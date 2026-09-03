from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evc.defense import (
    PosteriorBenefitMLPDetector,
    build_previous_step_inputs,
    canonical_state_scope,
    dae_reconstruction_with_history,
    defended_indices_for_scope,
    load_dae,
    load_detector,
    posterior_detector_probabilities,
    save_dae,
    save_dae_history,
    save_detector,
    save_detector_history,
    weighted_state_error_np,
)
from evc.merged_attacks import build_state_attacker
from evc.merged_core import (
    ATTACK_ALGORITHMS,
    ATTACK_DEFAULTS,
    ATTACK_SCENARIOS,
    ChargingEnv,
    Critic,
    DEFAULT_BASELINE_ACTOR_PATH,
    DEFAULT_BASELINE_BUNDLE_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SIGNALS_PATH,
    LEGACY_BASELINE_ACTOR_PATH,
    LEGACY_BASELINE_BUNDLE_PATH,
    POLICY_MODES,
    PROFILE_MAP,
    RewardProfile,
    canonical_attack_algorithm,
    ensure_dir,
    json_dump,
    load_actor_critic_bundle,
    load_actor_from_path,
    load_policy_actor,
    normalize_result_frame,
    prepare_device,
    resolve_default_baseline_bundle_path,
    resolve_max_duration_of_stay,
    save_actor,
    save_baseline_bundle,
    set_seed,
    split_csv_floats,
    split_csv_strings,
)
from evc.merged_pipeline import (
    DetectorDatasetBundle,
    PairDatasetBundle,
    build_pair_dataset_from_clean_trajectories,
    collect_clean_trajectories,
    evaluate_action_dataset,
    evaluate_rollout_bundle,
    get_arrivals,
    iter_evaluation_summaries,
    load_clean_trajectory_dataset,
    load_detector_dataset,
    load_pair_dataset,
    save_evaluation_bundle,
    save_clean_trajectory_dataset,
    save_detector_dataset,
    save_matrix_outputs,
    save_pair_dataset,
    save_train_history,
    select_detector_threshold,
    train_detector_from_bundle,
    train_agent,
    train_dae_from_bundle,
    rollout_episode,
)
from evc.offline_dae_det_temporal_shield import (
    LOCAL_SHIELD_INDICES,
    LocalTemporalShieldConfig,
    calibrate_local_temporal_shield,
    eval_temporal_shield_suite,
    load_temporal_shield_bundle,
    save_temporal_shield_bundle,
    tune_temporal_shield_with_attacks,
)
from evc.online_atla_ppo_lstm_sa import (
    evaluate_online_atla_ppo_lstm_sa_agent,
    load_online_atla_ppo_lstm_sa_bundle,
    save_atla_ppo_lstm_sa_eval,
    save_atla_ppo_lstm_sa_history,
    save_online_atla_ppo_lstm_sa_bundle,
    train_online_atla_ppo_lstm_sa_agent,
)
from evc.sa_ddpg import canonical_sa_train_attack, save_sa_ddpg_bundle, train_sa_ddpg_agent
from evc.atla_ddpg_learned import save_atla_ddpg_learned_bundle, train_atla_ddpg_learned_adversary_agent

from evc.wocar import save_wocar_bundle, train_wocar_agent
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CLEAN_DIR = PROJECT_ROOT / 'data' / 'dnormal'
DEFAULT_PAIR_DIR = PROJECT_ROOT / 'data' / 'dae'
DEFAULT_DETECTOR_DATA_DIR = PROJECT_ROOT / 'data' / 'detector'
DEFAULT_DAE_DIR = PROJECT_ROOT / 'models' / 'dae'
DEFAULT_DETECTOR_DIR = PROJECT_ROOT / 'models' / 'detector'
DEFAULT_SA_DDPG_DIR = PROJECT_ROOT / 'models' / 'sa_ddpg'
DEFAULT_ONLINE_PPO_LSTM_DIR = PROJECT_ROOT / 'models' / 'online_ppo_lstm'
DEFAULT_ONLINE_PPO_LSTM_BUNDLE = DEFAULT_ONLINE_PPO_LSTM_DIR / 'default_bundle.pt'
DEFAULT_ATLA_DIR = PROJECT_ROOT / 'models' / 'atla'
DEFAULT_ATLA_BUNDLE = DEFAULT_ATLA_DIR / 'atla_bundle.pt'
DEFAULT_ATLA_DDPG_DIR = PROJECT_ROOT / 'models' / 'atla_ddpg'
DEFAULT_ATLA_DDPG_BUNDLE = DEFAULT_ATLA_DDPG_DIR / 'atla_ddpg_bundle.pt'
DEFAULT_ONLINE_ATLA_PPO_LSTM_DIR = DEFAULT_ATLA_DIR
DEFAULT_ONLINE_ATLA_PPO_LSTM_LOCAL_BUNDLE = DEFAULT_ATLA_BUNDLE
DEFAULT_ONLINE_ATLA_PPO_LSTM_ALL_BUNDLE = DEFAULT_ATLA_BUNDLE
DEFAULT_ONLINE_ATLA_PPO_LSTM_SA_DIR = DEFAULT_ATLA_DIR
DEFAULT_ONLINE_ATLA_PPO_LSTM_SA_LOCAL_BUNDLE = DEFAULT_ATLA_BUNDLE
DEFAULT_ONLINE_ATLA_PPO_LSTM_SA_ALL_BUNDLE = DEFAULT_ATLA_BUNDLE
DEFAULT_OFFLINE_DAE_DET_TEMPORAL_SHIELD_DIR = PROJECT_ROOT / 'models' / 'offline_dae_det_temporal_shield'
DEFAULT_OFFLINE_DAE_DET_TEMPORAL_SHIELD_RESULTS_DIR = DEFAULT_RESULTS_DIR / 'offline_dae_det_temporal_shield'
DEFAULT_OFFLINE_SESSION_DAE_ADAPTIVE_DIR = PROJECT_ROOT / 'models' / 'offline_session_dae_adaptive'
DEFAULT_OFFLINE_SESSION_DAE_ADAPTIVE_PAIR_DIR = PROJECT_ROOT / 'data' / 'offline_session_dae_adaptive'
DEFAULT_OFFLINE_SESSION_DAE_ADAPTIVE_BALANCED_DIR = PROJECT_ROOT / 'models' / 'offline_session_dae_adaptive_balanced'
DEFAULT_OFFLINE_SESSION_DAE_ADAPTIVE_BALANCED_PAIR_DIR = PROJECT_ROOT / 'data' / 'offline_session_dae_adaptive_balanced'
DEFAULT_OFFLINE_SESSION_DETECTOR_DIR = PROJECT_ROOT / 'models' / 'offline_session_detector'
DEFAULT_WOCAR_DIR = PROJECT_ROOT / 'models' / 'wocar'
DEFAULT_UNIFIED_DIR = DEFAULT_RESULTS_DIR / 'unified_defense'
DEFAULT_FORMAL_MIN_CLEAN_SAMPLES = 2048
DEFAULT_FORMAL_MIN_CLEAN_GROUPS = 128
UNIFIED_SCOPE_DEFAULTS = {
    'local': {
        'dae_epochs': 25,
        'hidden_dim': 128,
        'latent_dim': 64,
        'decoder_hidden_dim': 128,
        'lambda_identity': 2.0,
        'lambda_robust': 0.1,
    },
    'all': {
        'dae_epochs': 30,
        'hidden_dim': 256,
        'latent_dim': 128,
        'decoder_hidden_dim': 256,
        'lambda_identity': 3.0,
        'lambda_robust': 0.2,
    },
}


def str2path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def ratio01(value: str) -> float:
    ratio = float(value)
    if ratio < 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError('attack ratio must be in [0, 1].')
    return ratio


def prefixed_attr(prefix: str, name: str) -> str:
    return f"{str(prefix or '').replace('-', '_')}{name}"


def _metadata_value_matches(actual, expected) -> bool:
    if isinstance(expected, (float, int, np.floating, np.integer)) and not isinstance(expected, bool):
        try:
            return bool(np.isclose(float(actual), float(expected), atol=1e-6, rtol=0.0))
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _read_torch_artifact_metadata(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if isinstance(payload, dict):
        return dict(payload.get('metadata') or {})
    return {}


def _expected_artifact_semantics(
    *,
    policy_tag: str,
    algorithm: str,
    scenario: str | None,
    epsilon: float,
    attack_ratio: float | None = None,
    attack_scope: str | None = None,
) -> dict:
    expected = {
        'policy_tag': str(policy_tag),
        'algorithm': str(algorithm),
        'scenario': canonical_attack_scenario(algorithm, scenario),
        'epsilon': float(epsilon),
    }
    if attack_ratio is not None:
        expected['attack_ratio'] = float(np.clip(float(attack_ratio), 0.0, 1.0))
    if attack_scope is not None:
        expected['attack_scope'] = str(attack_scope)
    return expected


def _validate_metadata(
    metadata: dict,
    *,
    artifact_kind: str,
    artifact_path: str | Path,
    expected: dict | None = None,
    required_values: dict | None = None,
) -> None:
    meta = dict(metadata or {})
    checks = {}
    if expected:
        checks.update(expected)
    if required_values:
        checks.update(required_values)
    mismatches: list[str] = []
    for key, expected_value in checks.items():
        if expected_value is None:
            continue
        actual_value = meta.get(key)
        if actual_value is None:
            if key == 'attack_ratio' and _metadata_value_matches(1.0, expected_value):
                continue
            if key == 'attack_scope' and _metadata_value_matches('obs', expected_value):
                continue
            mismatches.append(f'missing {key}={expected_value!r}')
            continue
        if not _metadata_value_matches(actual_value, expected_value):
            mismatches.append(f'{key} expected {expected_value!r} got {actual_value!r}')
    if mismatches:
        details = '; '.join(mismatches)
        raise ValueError(f'Incompatible {artifact_kind} metadata in {artifact_path}: {details}')


def set_all_seeds(seed: int) -> None:
    set_seed(seed)


def split_attack_state_scopes(text: str) -> list[str]:
    aliases = {
        'both': ['local', 'all'],
        'local+all': ['local', 'all'],
        'local_and_all': ['local', 'all'],
    }
    token = str(text or 'local,all').strip().lower()
    scopes = aliases.get(token)
    if scopes is None:
        scopes = split_csv_strings(token)
    out: list[str] = []
    for scope in scopes:
        if scope not in {'local', 'all'}:
            raise argparse.ArgumentTypeError(f'Unsupported ATLA-PPO-LSTM-SA attack state scope: {scope!r}')
        if scope not in out:
            out.append(scope)
    if not out:
        raise argparse.ArgumentTypeError('At least one attack state scope is required.')
    return out


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--data-path', type=str2path, default=DEFAULT_DATA_PATH, help='Path to arrivals CSV (data.csv).')
    parser.add_argument('--signals-path', type=str2path, default=DEFAULT_SIGNALS_PATH, help='Path to fixed signal profile file.')
    parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True, help='Prefer CUDA when available.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--num-threads', type=int, default=1, help='Limit PyTorch CPU threads.')
    parser.add_argument('--max-sessions', type=int, default=None, help='Use only first N sessions for quick validation.')


def add_actor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--policy-mode', choices=POLICY_MODES, default='baseline', help='Built-in policy source (currently baseline).')
    parser.add_argument('--actor-path', type=str2path, default=None, help='Explicit actor checkpoint path. Overrides policy-mode.')


def add_clean_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--clean-dir', type=str2path, default=DEFAULT_CLEAN_DIR, help='Directory for saved clean Dnormal rollouts.')
    parser.add_argument('--clean-path', type=str2path, default=None, help='Explicit Dnormal dataset path.')
    parser.add_argument('--clean-source', choices=['load', 'load_or_collect', 'collect'], default='load_or_collect', help='How to obtain the clean Dnormal rollout.')
    parser.add_argument(
        '--min-clean-samples',
        type=int,
        default=None,
        help='Minimum cached Dnormal sample count accepted before recollecting. Defaults to --collect-max-samples when available, otherwise 2048.',
    )
    parser.add_argument(
        '--min-clean-groups',
        type=int,
        default=None,
        help='Minimum cached Dnormal unique (episode, vehicle) groups accepted before recollecting. Defaults to an auto floor derived from the sample floor.',
    )


def add_attack_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--alpha', type=float, default=None, help='attack step size (optional override).')
    parser.add_argument('--iters', type=int, default=None, help='attack iterations (optional override).')
    parser.add_argument('--attack-ratio', type=ratio01, default=1.0, help='attack probability ratio in [0,1].')
    parser.add_argument('--attack-scope', choices=['obs', 'vehicle', 'window'], default='obs', help='partial attack scope.')
    parser.add_argument('--price-threshold', type=float, default=400.0, help='C-scenario price threshold.')
    parser.add_argument('--soc-new-threshold', type=float, default=0.5, help='O-scenario SOC threshold for new arrivals.')
    parser.add_argument('--soc-rollout-threshold', type=float, default=0.3, help='O-scenario SOC threshold for active vehicles.')
    parser.add_argument('--even-station-target', type=float, default=1.0, help='F-scenario target action for even station.')
    parser.add_argument('--odd-station-target', type=float, default=-0.5, help='F-scenario target action for odd station.')


def add_attack_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--algorithm', choices=ATTACK_ALGORITHMS, default='electhacker', help='Attack algorithm.')
    parser.add_argument('--scenario', choices=ATTACK_SCENARIOS, default='O', help='ElectHacker scenario (ignored by opposite/q-function attacks).')
    parser.add_argument('--epsilon', type=float, default=None, help='Attack epsilon. Uses default if omitted.')
    parser.add_argument('--attack-bundle-path', type=str2path, default=None, help='Optional actor+critic bundle for q_function attack.')
    add_attack_tuning_args(parser)


def add_dae_model_args(parser: argparse.ArgumentParser, *, prefix: str = '') -> None:
    parser.add_argument(f'--{prefix}seq-len', type=int, default=8, help='Sequence window size for the GRU-VAE denoiser.')
    parser.add_argument(f'--{prefix}hidden-dim', type=int, default=128, help='GRU hidden size for the GRU-VAE denoiser.')
    parser.add_argument(f'--{prefix}latent-dim', type=int, default=64, help='Latent size for the GRU-VAE denoiser.')
    parser.add_argument(f'--{prefix}num-layers', type=int, default=1, help='Number of GRU layers for the GRU-VAE denoiser.')
    parser.add_argument(f'--{prefix}decoder-hidden-dim', type=int, default=128, help='Decoder hidden size for the GRU-VAE denoiser.')
    parser.add_argument(f'--{prefix}beta-kl', type=float, default=1e-4, help='KL weight in the GRU-VAE objective.')
    parser.add_argument(f'--{prefix}lambda-robust', type=float, default=0.1, help='Deterministic policy robustness regularizer weight.')
    parser.add_argument(f'--{prefix}include-clean-sequences', action=argparse.BooleanOptionalAction, default=True, help='Mix clean Dnormal sequences into the GRU-VAE training set (paper-style Dadv -> Dnormal).')


def add_detector_selection_args(parser: argparse.ArgumentParser, *, prefix: str = '', default_grid_size: int = 15) -> None:
    parser.add_argument(f'--{prefix}detector-clean-reward-floor-ratio', type=float, default=0.95, help='During threshold search, require at least this fraction of clean baseline reward before preferring higher defended attack reward.')
    parser.add_argument(f'--{prefix}detector-threshold-grid-size', type=int, default=int(default_grid_size), help='Number of detector threshold candidates for rollout selection.')


def resolve_actor(args, device: torch.device):
    if getattr(args, 'actor_path', None) is not None:
        return load_actor_from_path(args.actor_path, device)
    return load_policy_actor(args.policy_mode, device)


def default_eval_dae_load_path(args, policy_tag: str) -> Path | None:
    if getattr(args, 'dae_path', None) is not None:
        return None
    wants_dae = any(
        value is not None
        for value in (
            getattr(args, 'detector_path', None),
            getattr(args, 'detector_threshold', None),
        )
    )
    if not wants_dae:
        return None
    return default_dae_path(
        policy_tag,
        args.algorithm,
        args.scenario,
        effective_attack_epsilon(args),
        DEFAULT_DAE_DIR,
        attack_ratio=args.attack_ratio,
        attack_scope=args.attack_scope,
    )


def default_eval_detector_load_path(args, policy_tag: str) -> Path | None:
    if getattr(args, 'detector_path', None) is not None:
        return None
    wants_detector = getattr(args, 'detector_threshold', None) is not None
    if not wants_detector:
        return None
    return default_detector_path(
        policy_tag,
        args.algorithm,
        args.scenario,
        effective_attack_epsilon(args),
        DEFAULT_DETECTOR_DIR,
        attack_ratio=args.attack_ratio,
        attack_scope=args.attack_scope,
        detector_feature_mode=getattr(args, 'detector_feature_mode', 'sequence'),
    )


def resolve_dae(args, device: torch.device, *, expected_metadata: dict | None = None, default_path: Path | None = None):
    dae_path = getattr(args, 'dae_path', None) or default_path
    if dae_path is None:
        return None
    metadata = _read_torch_artifact_metadata(dae_path)
    if expected_metadata is not None:
        _validate_metadata(
            metadata,
            artifact_kind='DAE artifact',
            artifact_path=dae_path,
            expected=expected_metadata,
        )
    return load_dae(dae_path, device)


def effective_attack_epsilon(args) -> float:
    if getattr(args, 'epsilon', None) is not None:
        return float(args.epsilon)
    return float(ATTACK_DEFAULTS[canonical_attack_algorithm(str(getattr(args, 'algorithm', 'electhacker')))].epsilon)


def scenario_applies(algorithm: str) -> bool:
    return canonical_attack_algorithm(str(algorithm)) == 'electhacker'


def canonical_attack_scenario(algorithm: str, scenario: str | None) -> str:
    return str(scenario or 'O') if scenario_applies(algorithm) else 'O'


def attack_name_fragment(algorithm: str, scenario: str | None) -> str:
    algorithm_name = canonical_attack_algorithm(str(algorithm))
    if scenario_applies(algorithm_name):
        return f'{algorithm_name}_{canonical_attack_scenario(algorithm_name, scenario)}'
    return algorithm_name


def resolve_attack_bundle_path(args) -> Path | None:
    explicit = getattr(args, 'attack_bundle_path', None)
    if explicit is not None:
        return Path(explicit)
    actor_path = getattr(args, 'actor_path', None)
    if actor_path is not None and _torch_artifact_has_critic(actor_path):
        return Path(actor_path)
    if getattr(args, 'actor_path', None) is None:
        candidate = resolve_default_baseline_bundle_path()
        if Path(candidate).exists():
            return Path(candidate)
    return None


def _torch_artifact_has_critic(path: str | Path | None) -> bool:
    if path is None:
        return False
    try:
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get('critic_state_dict') is not None


def load_attack_critic(algorithm: str, device: torch.device, *, bundle_path: Path | None) -> Critic | None:
    if canonical_attack_algorithm(algorithm) != 'q_function':
        return None
    if bundle_path is None:
        raise ValueError('q_function attack requires --attack-bundle-path when actor-path does not map to the default baseline bundle.')
    payload = load_actor_critic_bundle(bundle_path, device)
    critic_state = payload.get('critic_state_dict')
    if critic_state is None:
        raise ValueError(f'Attack bundle does not contain critic weights: {bundle_path}')
    critic = Critic().to(device)
    critic.load_state_dict(critic_state)
    critic.eval()
    return critic


def default_temporal_shield_bundle_path(state_scope: str, out_dir: Path | None = None) -> Path:
    scope = canonical_state_scope(state_scope)
    root = Path(out_dir or DEFAULT_OFFLINE_DAE_DET_TEMPORAL_SHIELD_DIR)
    return root / f'default_{scope}_temporal_shield.pt'


def default_temporal_shield_dae_path(state_scope: str) -> Path:
    scope = canonical_state_scope(state_scope)
    name = 'dae_baseline_paper_opposite_pgd_q_function_eps0p1_scopeall.pt' if scope == 'all' else 'dae_baseline_paper_opposite_pgd_q_function_eps0p1.pt'
    return DEFAULT_DAE_DIR / name


def default_temporal_shield_detector_path(state_scope: str) -> Path:
    scope = canonical_state_scope(state_scope)
    name = 'detector_baseline_paper_opposite_pgd_q_function_eps0p1_scopeall_post.pt' if scope == 'all' else 'detector_baseline_paper_opposite_pgd_q_function_eps0p1_post.pt'
    return DEFAULT_DETECTOR_DIR / name


def resolve_temporal_shield_actor(args, device: torch.device):
    return resolve_actor(args, device)


def temporal_shield_safe_recovery(clean_reward: float, attack_reward: float, defended_reward: float) -> float:
    denom = float(clean_reward - attack_reward)
    if abs(denom) <= 1e-8:
        return 0.0
    return float((defended_reward - attack_reward) / denom)


def default_clean_path(policy_tag: str, reward_profile: str, clean_dir: Path | None = None) -> Path:
    root = Path(clean_dir or DEFAULT_CLEAN_DIR)
    return root / f'dnormal_{policy_tag}_{str(reward_profile)}.npz'


def _clean_group_count(bundle) -> int:
    if getattr(bundle, 'episode_indices', None) is None or getattr(bundle, 'vehicle_ids', None) is None:
        return 0
    episode_indices = np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1)
    vehicle_ids = np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1)
    return len({(int(e), int(v)) for e, v in zip(episode_indices.tolist(), vehicle_ids.tolist())})


def _resolve_clean_requirements(args, *, max_samples: int | None) -> tuple[int, int]:
    raw_min_samples = getattr(args, 'min_clean_samples', None)
    raw_min_groups = getattr(args, 'min_clean_groups', None)
    if raw_min_samples is None:
        min_clean_samples = int(max_samples) if max_samples is not None else DEFAULT_FORMAL_MIN_CLEAN_SAMPLES
    else:
        min_clean_samples = int(raw_min_samples)
    min_clean_samples = max(min_clean_samples, 0)
    if raw_min_groups is None:
        if min_clean_samples <= 0:
            min_clean_groups = 0
        else:
            auto_group_floor = max(1, min_clean_samples // 16)
            min_clean_groups = min(DEFAULT_FORMAL_MIN_CLEAN_GROUPS, auto_group_floor)
    else:
        min_clean_groups = int(raw_min_groups)
    min_clean_groups = max(min_clean_groups, 0)
    return int(min_clean_samples), int(min_clean_groups)


def load_or_collect_clean_bundle(
    args,
    arrivals: pd.DataFrame,
    actor,
    device: torch.device,
    *,
    reward_profile: str,
    policy_tag: str,
    episodes: int,
    max_samples: int | None,
):
    clean_path = getattr(args, 'clean_path', None) or default_clean_path(policy_tag, reward_profile, getattr(args, 'clean_dir', None))
    clean_source = str(getattr(args, 'clean_source', 'load_or_collect'))
    min_clean_samples, min_clean_groups = _resolve_clean_requirements(args, max_samples=max_samples)
    if clean_source in {'load', 'load_or_collect'} and Path(clean_path).exists():
        bundle = load_clean_trajectory_dataset(clean_path)
        sample_count = int(np.asarray(bundle.clean_inputs, dtype=np.float32).shape[0])
        group_count = _clean_group_count(bundle)
        if sample_count >= min_clean_samples and group_count >= min_clean_groups:
            return bundle, Path(clean_path), 'loaded'
        if clean_source == 'load':
            raise RuntimeError(f'Loaded clean Dnormal dataset is too small: samples={sample_count}, groups={group_count}, required>={min_clean_samples}/{min_clean_groups}')
        print(f'cached Dnormal too small, recollecting: path={clean_path} samples={sample_count} groups={group_count} required>={min_clean_samples}/{min_clean_groups}')
    elif clean_source == 'load':
        raise FileNotFoundError(f'Missing clean Dnormal dataset: {clean_path}')
    bundle = collect_clean_trajectories(
        arrivals,
        actor,
        args.signals_path,
        device,
        reward_profile=PROFILE_MAP[reward_profile],
        episodes=episodes,
        max_samples=max_samples,
    )
    save_clean_trajectory_dataset(bundle, clean_path)
    return bundle, Path(clean_path), 'collected'


def actor_source_tag(policy_mode: str, actor_path: Path | None) -> str:
    return policy_mode if actor_path is None else Path(actor_path).stem


def dae_usage_tag(dae_path: Path | None) -> str:
    return 'with_dae' if dae_path is not None else 'no_dae'


def rollout_usage_tag(
    dae_path: Path | None,
    detector_path: Path | None = None,
    detector_threshold: float | None = None,
) -> str:
    has_detector = detector_path is not None or detector_threshold is not None
    if dae_path is None:
        return 'no_dae'
    if has_detector:
        return 'with_dae_detector'
    return 'with_dae'


def result_dir_for_rollout(
    base_dir: Path,
    *,
    policy_tag: str,
    attack_tag: str,
    dae_tag: str,
    epsilon: float,
    reward_profile: str,
    attack_ratio: float = 1.0,
    attack_scope: str = 'obs',
) -> Path:
    cfg = attack_config_fragment(attack_ratio, attack_scope)
    path = Path(base_dir) / f'rollout__{policy_tag}__{attack_tag}__{dae_tag}__eps{float(epsilon):g}__{reward_profile}{cfg}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_dir_for_actions(
    base_dir: Path,
    *,
    policy_tag: str,
    attack_tag: str,
    dae_tag: str,
    epsilon: float,
    source_tag: str,
    attack_ratio: float = 1.0,
    attack_scope: str = 'obs',
) -> Path:
    cfg = attack_config_fragment(attack_ratio, attack_scope)
    token = cfg.lstrip('_')
    source = str(source_tag)
    suffix = '' if (token and source.endswith(token)) else cfg
    path = Path(base_dir) / f'actions__{policy_tag}__{attack_tag}__{dae_tag}__eps{float(epsilon):g}__src{source}{suffix}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_baseline_actor_save_path(
    output_dir: Path,
    actor_model_name: str | None,
    episodes: int,
    seed: int,
    reward_profile: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if actor_model_name:
        return output_dir / actor_model_name
    reward_suffix = ''
    if reward_profile is not None and str(reward_profile) != 'train':
        reward_suffix = f'_{_filename_token(reward_profile)}'
    return output_dir / f'actor_baseline{reward_suffix}_ep{int(episodes)}_seed{int(seed)}.pt'


def resolve_baseline_bundle_save_path(
    output_dir: Path,
    bundle_name: str | None,
    episodes: int,
    seed: int,
    reward_profile: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle_name:
        return output_dir / bundle_name
    reward_suffix = ''
    if reward_profile is not None and str(reward_profile) != 'train':
        reward_suffix = f'_{_filename_token(reward_profile)}'
    return output_dir / f'baseline_bundle{reward_suffix}_ep{int(episodes)}_seed{int(seed)}.pt'


def resolve_sa_ddpg_actor_save_path(output_dir: Path, actor_model_name: str | None, episodes: int, seed: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if actor_model_name:
        return output_dir / actor_model_name
    return output_dir / f'actor_sa_ddpg_ep{int(episodes)}_seed{int(seed)}.pt'


def resolve_atla_ddpg_actor_save_path(output_dir: Path, actor_model_name: str | None, episodes: int, seed: int, state_scope: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if actor_model_name:
        return output_dir / actor_model_name
    scope = _filename_token(canonical_state_scope(state_scope))
    return output_dir / f'actor_atla_ddpg_ep{int(episodes)}_seed{int(seed)}_{scope}.pt'


def _filename_token(value: object, *, default: str = 'mixed') -> str:
    token = str(value or default).strip().lower().replace('-', '_')
    token = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in token)
    token = '_'.join(part for part in token.split('_') if part)
    return token or default


def _attack_filename_token(attacks: list[str] | tuple[str, ...] | str | None) -> str:
    tokens = split_csv_strings(attacks) if isinstance(attacks, str) else list(attacks or [])
    if not tokens:
        return 'mixed'
    aliases = {
        'opposite_pgd': 'pgd',
        'pgd': 'pgd',
        'q_function': 'q',
        'q': 'q',
        'q_function_attack': 'q',
        'opposite_fgsm': 'fgsm',
        'fgsm': 'fgsm',
        'electhacker_o': 'electhacker_o',
        'electhacker': 'electhacker_o',
    }
    return '_'.join(aliases.get(_filename_token(token), _filename_token(token)) for token in tokens)


def _float_filename_token(value: float) -> str:
    token = f'{float(value):g}'.replace('-', 'm').replace('.', 'p')
    return _filename_token(token, default='0')


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    h = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def attack_config_fragment(attack_ratio: float, attack_scope: str) -> str:
    ratio = float(np.clip(float(attack_ratio), 0.0, 1.0))
    scope = str(attack_scope).strip().lower() or 'obs'
    if abs(ratio - 1.0) < 1e-12 and scope == 'obs':
        return ''
    ratio_text = f'{ratio:g}'.replace('.', 'p')
    return f'_ar{ratio_text}_scope{scope}'


def output_dir_for_attack_config(base_dir: str | Path, *, attack_ratio: float, attack_scope: str) -> Path:
    root = Path(base_dir)
    cfg = attack_config_fragment(attack_ratio, attack_scope)
    if not cfg:
        return root
    token = cfg.lstrip('_')
    if root.name.endswith(token):
        return root
    return root / token


def default_pair_path(
    policy_tag: str,
    algorithm: str,
    scenario: str | None,
    epsilon: float,
    pair_dir: Path | None = None,
    *,
    attack_ratio: float = 1.0,
    attack_scope: str = 'obs',
) -> Path:
    root = Path(pair_dir or DEFAULT_PAIR_DIR)
    cfg = attack_config_fragment(attack_ratio, attack_scope)
    return root / f'dae_pairs_{policy_tag}_attacked_{attack_name_fragment(algorithm, scenario)}_eps{float(epsilon):g}{cfg}.npz'


def default_dae_path(
    policy_tag: str,
    algorithm: str,
    scenario: str | None,
    epsilon: float,
    dae_dir: Path | None = None,
    *,
    attack_ratio: float = 1.0,
    attack_scope: str = 'obs',
) -> Path:
    root = Path(dae_dir or DEFAULT_DAE_DIR)
    cfg = attack_config_fragment(attack_ratio, attack_scope)
    return root / f'dae_{policy_tag}_{attack_name_fragment(algorithm, scenario)}_eps{float(epsilon):g}{cfg}.pt'


def _safe_tag_token(value: str) -> str:
    return str(value).strip().lower().replace('-', '_').replace('.', 'p').replace(',', '_')


def unified_profile_tag(algorithms: list[str], epsilons: list[float], *, prefix: str = 'paper') -> str:
    alg_tokens = [_safe_tag_token(attack_name_fragment(algorithm, None)) for algorithm in algorithms]
    deduped_algs = list(dict.fromkeys(alg_tokens))
    eps_tokens = [_safe_tag_token(f'{float(epsilon):g}') for epsilon in epsilons]
    deduped_eps = list(dict.fromkeys(eps_tokens))
    return f"{_safe_tag_token(prefix)}_{'_'.join(deduped_algs)}_eps{'_'.join(deduped_eps)}"


def default_unified_pair_path(policy_tag: str, profile_tag: str, pair_dir: Path | None = None) -> Path:
    root = Path(pair_dir or DEFAULT_PAIR_DIR)
    return root / f'dadv_{policy_tag}_{profile_tag}.npz'


def default_unified_dae_path(policy_tag: str, profile_tag: str, dae_dir: Path | None = None) -> Path:
    root = Path(dae_dir or DEFAULT_DAE_DIR)
    return root / f'dae_{policy_tag}_{profile_tag}.pt'


def default_unified_detector_dataset_path(
    policy_tag: str,
    profile_tag: str,
    detector_data_dir: Path | None = None,
    *,
    detector_feature_mode: str = 'posterior',
    posterior_label_mode: str | None = 'benefit',
) -> Path:
    root = Path(detector_data_dir or DEFAULT_DETECTOR_DATA_DIR)
    return root / f'detector_dataset_{policy_tag}_{profile_tag}{detector_feature_suffix(detector_feature_mode, posterior_label_mode)}.npz'


def default_unified_detector_path(
    policy_tag: str,
    profile_tag: str,
    detector_dir: Path | None = None,
    *,
    detector_feature_mode: str = 'posterior',
    posterior_label_mode: str | None = 'benefit',
) -> Path:
    root = Path(detector_dir or DEFAULT_DETECTOR_DIR)
    return root / f"detector_{policy_tag}_{profile_tag}{detector_feature_suffix(detector_feature_mode, posterior_label_mode)}.pt"



def _scope_matches_model(model, state_scope: str) -> bool:
    expected = tuple(defended_indices_for_scope(state_scope))
    actual = tuple(int(v) for v in getattr(model, 'local_indices', expected))
    return actual == expected


def _validate_state_scope_alignment(label: str, path: str | Path, metadata: dict | None, state_scope: str) -> None:
    actual = canonical_state_scope((metadata or {}).get('state_scope', 'local'))
    expected = canonical_state_scope(state_scope)
    if actual != expected:
        raise ValueError(f'{label} state_scope mismatch for {path}: expected {expected!r}, got {actual!r}.')


def apply_unified_scope_defaults(args, state_scope: str) -> dict[str, float | int]:
    scope = canonical_state_scope(state_scope)
    local_defaults = UNIFIED_SCOPE_DEFAULTS['local']
    scope_defaults = UNIFIED_SCOPE_DEFAULTS[scope]
    applied: dict[str, float | int] = {}
    for name, value in scope_defaults.items():
        current = getattr(args, name, None)
        if current is None or current == local_defaults[name]:
            setattr(args, name, value)
            applied[name] = value
    return applied


def default_named_detector_path(
    artifact_tag: str,
    policy_tag: str,
    algorithm: str,
    scenario: str | None,
    epsilon: float,
    detector_dir: Path | None = None,
    *,
    attack_ratio: float = 1.0,
    attack_scope: str = 'obs',
    detector_feature_mode: str = 'sequence',
) -> Path:
    root = Path(detector_dir or DEFAULT_DETECTOR_DIR)
    cfg = attack_config_fragment(attack_ratio, attack_scope)
    feature_suffix = detector_feature_suffix(detector_feature_mode)
    return root / f'{artifact_tag}_{policy_tag}_{attack_name_fragment(algorithm, scenario)}_eps{float(epsilon):g}{cfg}{feature_suffix}.pt'


def default_detector_path(
    policy_tag: str,
    algorithm: str,
    scenario: str | None,
    epsilon: float,
    detector_dir: Path | None = None,
    *,
    attack_ratio: float = 1.0,
    attack_scope: str = 'obs',
    detector_feature_mode: str = 'sequence',
) -> Path:
    return default_named_detector_path(
        'detector',
        policy_tag,
        algorithm,
        scenario,
        epsilon,
        detector_dir,
        attack_ratio=attack_ratio,
        attack_scope=attack_scope,
        detector_feature_mode=detector_feature_mode,
    )


def maybe_publish(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def publish_best_baseline_artifacts(actor_src: Path, bundle_src: Path | None = None) -> None:
    actor_targets = [Path(DEFAULT_BASELINE_ACTOR_PATH), Path(LEGACY_BASELINE_ACTOR_PATH)]
    bundle_targets = [Path(DEFAULT_BASELINE_BUNDLE_PATH), Path(LEGACY_BASELINE_BUNDLE_PATH)]
    for target in actor_targets:
        maybe_publish(actor_src, target)
    if bundle_src is not None:
        for target in bundle_targets:
            maybe_publish(bundle_src, target)


def build_attacker(
    actor,
    device: torch.device,
    *,
    algorithm: str,
    epsilon: float,
    alpha: float | None,
    iters: int | None,
    seed: int,
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
    critic: Critic | None = None,
    attack_state_scope: str = 'local',
    signals_path: Path | None = None,
    reward_profile='train',
):
    return build_state_attacker(
        actor,
        device=device,
        algorithm=algorithm,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        seed=seed,
        obs_low=obs_low,
        obs_high=obs_high,
        critic=critic,
        attack_state_scope=attack_state_scope,
        signals_path=signals_path or DEFAULT_SIGNALS_PATH,
        reward_profile=reward_profile if isinstance(reward_profile, RewardProfile) else PROFILE_MAP[str(reward_profile)],
    )


def resolve_attack_obs_bounds(arrivals: pd.DataFrame, signals_path: Path) -> tuple[np.ndarray, np.ndarray]:
    env = ChargingEnv(signals_path=signals_path, reward_profile=PROFILE_MAP['train'])
    max_duration = resolve_max_duration_of_stay(arrivals)
    return env.observation_bounds(max_duration_of_stay=max_duration)


def build_attacker_from_args(args, actor, device: torch.device, *, arrivals: pd.DataFrame | None = None):
    resolved_arrivals = arrivals
    if resolved_arrivals is None:
        resolved_arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    obs_low, obs_high = resolve_attack_obs_bounds(resolved_arrivals, args.signals_path)
    critic = load_attack_critic(args.algorithm, device, bundle_path=resolve_attack_bundle_path(args))
    return build_attacker(
        actor,
        device,
        algorithm=args.algorithm,
        epsilon=effective_attack_epsilon(args),
        alpha=args.alpha,
        iters=args.iters,
        seed=args.seed,
        obs_low=obs_low,
        obs_high=obs_high,
        critic=critic,
        attack_state_scope=getattr(args, 'state_scope', 'local'),
        signals_path=args.signals_path,
        reward_profile='train',
    )


def iter_attack_jobs(algorithms: list[str], scenarios: list[str], epsilons: list[float]):
    for epsilon in epsilons:
        for algorithm in algorithms:
            scenario_iter = scenarios if scenario_applies(algorithm) else [None]
            for scenario in scenario_iter:
                canonical_scenario = canonical_attack_scenario(algorithm, scenario)
                yield {
                    'algorithm': algorithm,
                    'scenario': canonical_scenario if scenario_applies(algorithm) else None,
                    'epsilon': float(epsilon),
                    'attack_tag': attack_name_fragment(algorithm, canonical_scenario),
                }


def collect_pair_bundle_for_job(
    arrivals: pd.DataFrame,
    actor,
    device: torch.device,
    signals_path: Path,
    *,
    algorithm: str,
    scenario: str | None,
    epsilon: float,
    seed: int,
    alpha: float | None,
    iters: int | None,
    reward_profile: str,
    episodes: int,
    max_samples: int | None,
    policy_input_mode: str,
    price_threshold: float,
    soc_new_threshold: float,
    soc_rollout_threshold: float,
    even_station_target: float,
    odd_station_target: float,
    attack_ratio: float,
    attack_scope: str,
    state_scope: str = 'local',
    clean_bundle=None,
    critic: Critic | None = None,
):
    state_scope = canonical_state_scope(state_scope)
    obs_low, obs_high = resolve_attack_obs_bounds(arrivals, signals_path)
    attacker = build_attacker(
        actor,
        device,
        algorithm=algorithm,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        seed=seed,
        obs_low=obs_low,
        obs_high=obs_high,
        critic=critic,
        attack_state_scope=state_scope,
    )
    if clean_bundle is None:
        raise ValueError('Unified pair construction requires an explicit Dnormal clean_bundle.')
    bundle = build_pair_dataset_from_clean_trajectories(
        clean_bundle,
        attacker,
        canonical_attack_scenario(algorithm, scenario),
        price_threshold=price_threshold,
        soc_new_threshold=soc_new_threshold,
        soc_rollout_threshold=soc_rollout_threshold,
        even_station_target=even_station_target,
        odd_station_target=odd_station_target,
        attack_ratio=attack_ratio,
        attack_scope=attack_scope,
    )
    bundle.metadata.update(
        {
            'scenario': canonical_attack_scenario(algorithm, scenario),
            'reward_profile': str(reward_profile),
            'requested_policy_input_mode': str(policy_input_mode),
            'clean_collection_samples': int(np.asarray(clean_bundle.clean_inputs, dtype=np.float32).shape[0]),
            'time_indices_present': True,
            'detector_context_present': True,
            'state_scope': state_scope,
            'attack_state_scope': state_scope,
            'defense_state_scope': state_scope,
            'attack_state_indices': list(defended_indices_for_scope(state_scope)),
            'attack_trigger_mode': f'candidate_all_{state_scope}_obs',
        }
    )
    return bundle, attacker


def _episode_offset_for_bundle(bundle: PairDatasetBundle | DetectorDatasetBundle) -> int:
    episodes = np.asarray(getattr(bundle, 'episode_indices', np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    return int(np.max(episodes)) + 1 if episodes.size else 1


def merge_pair_bundles_for_unified(bundles: list[PairDatasetBundle], *, attack_tags: list[str]) -> PairDatasetBundle:
    if not bundles:
        raise ValueError('Unified DAE training requires at least one pair bundle.')
    source_scopes = [str((bundle.metadata or {}).get('state_scope', 'local')) for bundle in bundles]
    state_scope = canonical_state_scope(source_scopes[0])
    if any(canonical_state_scope(scope) != state_scope for scope in source_scopes):
        raise ValueError(f'Unified pair merge requires one state_scope, got {source_scopes!r}.')
    adv_inputs: list[np.ndarray] = []
    clean_inputs: list[np.ndarray] = []
    time_indices: list[np.ndarray] = []
    stations: list[np.ndarray] = []
    is_new_arrivals: list[np.ndarray] = []
    vehicle_ids: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    attack_masks: list[np.ndarray] = []
    episode_offset = 0
    for idx, bundle in enumerate(bundles):
        adv = np.asarray(bundle.adv_inputs, dtype=np.float32).reshape(-1, 11)
        clean = np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
        count = int(clean.shape[0])
        if adv.shape != clean.shape:
            raise ValueError('Unified pair merge requires aligned adv_inputs and clean_inputs.')
        adv_inputs.append(adv)
        clean_inputs.append(clean)
        time_indices.append(np.asarray(bundle.time_indices, dtype=np.int64).reshape(-1) if bundle.time_indices is not None else np.zeros((count,), dtype=np.int64))
        stations.append(np.asarray(bundle.stations, dtype=np.int64).reshape(-1) if bundle.stations is not None else np.zeros((count,), dtype=np.int64))
        is_new_arrivals.append(np.asarray(bundle.is_new_arrivals, dtype=np.int64).reshape(-1) if bundle.is_new_arrivals is not None else np.zeros((count,), dtype=np.int64))
        vehicle_ids.append(np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1) if bundle.vehicle_ids is not None else np.arange(count, dtype=np.int64))
        raw_episode = np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1) if bundle.episode_indices is not None else np.zeros((count,), dtype=np.int64)
        episode_indices.append(raw_episode + int(episode_offset))
        attack_masks.append(np.asarray(bundle.attack_mask, dtype=np.int64).reshape(-1) if bundle.attack_mask is not None else (np.max(np.abs(adv - clean), axis=1) > 1e-8).astype(np.int64))
        episode_offset += max(_episode_offset_for_bundle(bundle), 1)
    merged_metadata = {
        'collection_mode': 'unified_offline_attack_from_dnormal',
        'train_attacks': list(attack_tags),
        'samples': int(sum(arr.shape[0] for arr in clean_inputs)),
        'attacked_samples': int(sum(int(mask.sum()) for mask in attack_masks)),
        'source_bundle_count': int(len(bundles)),
        'policy_input_mode': 'clean',
        'attack_trigger_mode': f'candidate_all_{state_scope}_obs',
        'state_scope': state_scope,
        'attack_state_scope': state_scope,
        'defense_state_scope': state_scope,
        'attack_state_indices': list(defended_indices_for_scope(state_scope)),
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


def detector_dataset_from_unified_pair(bundle: PairDatasetBundle, *, profile_tag: str, train_attack_tags: list[str], state_scope: str = 'local') -> DetectorDatasetBundle:
    state_scope = canonical_state_scope(state_scope)
    clean_inputs = np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
    adv_inputs = np.asarray(bundle.adv_inputs, dtype=np.float32).reshape(-1, 11)
    count = int(clean_inputs.shape[0])
    metadata = {
        'collection_mode': 'unified_detector_clean_plus_offline_eval',
        'detector_mode': 'pre',
        'profile_tag': str(profile_tag),
        'train_attacks': list(train_attack_tags),
        'samples': count,
        'attacked_samples': int(np.sum(np.asarray(bundle.attack_mask, dtype=np.int64).reshape(-1))) if bundle.attack_mask is not None else int(np.sum(np.max(np.abs(adv_inputs - clean_inputs), axis=1) > 1e-8)),
        'policy_input_mode': 'clean',
        'attack_trigger_mode': f'candidate_all_{state_scope}_obs',
        'state_scope': state_scope,
        'attack_state_scope': state_scope,
        'defense_state_scope': state_scope,
        'attack_state_indices': list(defended_indices_for_scope(state_scope)),
    }
    return DetectorDatasetBundle(
        clean_inputs=clean_inputs,
        adv_inputs=adv_inputs,
        metadata=metadata,
        time_indices=np.asarray(bundle.time_indices, dtype=np.int64).reshape(-1) if bundle.time_indices is not None else np.zeros((count,), dtype=np.int64),
        stations=np.asarray(bundle.stations, dtype=np.int64).reshape(-1) if bundle.stations is not None else np.zeros((count,), dtype=np.int64),
        is_new_arrivals=np.asarray(bundle.is_new_arrivals, dtype=np.int64).reshape(-1) if bundle.is_new_arrivals is not None else np.zeros((count,), dtype=np.int64),
        vehicle_ids=np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1) if bundle.vehicle_ids is not None else np.arange(count, dtype=np.int64),
        episode_indices=np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1) if bundle.episode_indices is not None else np.zeros((count,), dtype=np.int64),
        attack_mask=None if bundle.attack_mask is None else np.asarray(bundle.attack_mask, dtype=np.int64).reshape(-1),
    )


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
    posterior_label_mode: str = 'benefit',
    use_benefit_sample_weights: bool = True,
    state_scope: str = 'local',
    repair_mode: str = 'full',
) -> DetectorDatasetBundle:
    if defender is None:
        raise ValueError('posterior detector dataset requires a trained or loaded DAE defender.')
    label_mode = canonical_posterior_label_mode(posterior_label_mode)
    state_scope = canonical_state_scope(state_scope)
    repair_mode = str(repair_mode or 'full').strip().lower().replace('-', '_')
    if repair_mode not in {'full', 'core_only'}:
        raise ValueError(f'Unsupported posterior repair_mode: {repair_mode!r}')
    # IMPORTANT: posterior labels must measure the candidate state that is
    # actually passed to the policy when DET routes to Denoise.  In the final
    # core-only DTSR runtime the DAE reconstructs all 11 dimensions, but only
    # SOC / remaining-time / cumulative-cost are injected into the policy
    # observation.  Using the full reconstruction here would train DET against
    # a different action candidate than runtime and can invert benefit labels.
    state_indices = tuple(int(v) for v in (LOCAL_SHIELD_INDICES if repair_mode == 'core_only' else defended_indices_for_scope(state_scope)))
    clean_inputs = np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
    adv_inputs = np.asarray(bundle.adv_inputs, dtype=np.float32).reshape(-1, 11)
    attack_mask = (
        (np.max(np.abs(adv_inputs - clean_inputs), axis=1) > 1e-8).astype(np.int64)
        if bundle.attack_mask is None
        else np.asarray(bundle.attack_mask, dtype=np.int64).reshape(-1)
    )
    time_indices = np.asarray(bundle.time_indices, dtype=np.int64).reshape(-1) if bundle.time_indices is not None else np.zeros((clean_inputs.shape[0],), dtype=np.int64)
    stations = np.asarray(bundle.stations, dtype=np.int64).reshape(-1) if bundle.stations is not None else np.zeros((clean_inputs.shape[0],), dtype=np.int64)
    is_new_arrivals = np.asarray(bundle.is_new_arrivals, dtype=np.int64).reshape(-1) if bundle.is_new_arrivals is not None else np.zeros((clean_inputs.shape[0],), dtype=np.int64)
    vehicle_ids = np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1) if bundle.vehicle_ids is not None else np.arange(clean_inputs.shape[0], dtype=np.int64)
    episode_indices = np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1) if bundle.episode_indices is not None else np.zeros((clean_inputs.shape[0],), dtype=np.int64)

    adv_prev = build_previous_step_inputs(adv_inputs, episode_indices=episode_indices, vehicle_ids=vehicle_ids)
    clean_prev = build_previous_step_inputs(clean_inputs, episode_indices=episode_indices, vehicle_ids=vehicle_ids)
    adv_rec_full = dae_reconstruction_with_history(defender, adv_inputs, device, episode_indices=episode_indices, vehicle_ids=vehicle_ids)
    clean_rec_full = dae_reconstruction_with_history(defender, clean_inputs, device, episode_indices=episode_indices, vehicle_ids=vehicle_ids)
    if repair_mode == 'core_only':
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
        clean_t = torch.as_tensor(clean_inputs, dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv_inputs, dtype=torch.float32, device=device)
        adv_rec_t = torch.as_tensor(adv_rec, dtype=torch.float32, device=device)
        clean_rec_t = torch.as_tensor(clean_rec, dtype=torch.float32, device=device)
        clean_act = actor(clean_t).detach().cpu().numpy().astype(np.float32).reshape(-1)
        adv_act = actor(adv_t).detach().cpu().numpy().astype(np.float32).reshape(-1)
        adv_rec_act = actor(adv_rec_t).detach().cpu().numpy().astype(np.float32).reshape(-1)
        clean_rec_act = actor(clean_rec_t).detach().cpu().numpy().astype(np.float32).reshape(-1)

    action_weight = max(float(benefit_action_weight), 0.0)
    state_weight = max(float(benefit_state_weight), 0.0)
    margin = max(float(benefit_margin), 0.0)
    adv_benefit = (
        state_weight * (
            weighted_state_error_np(adv_inputs, clean_inputs, state_indices=state_indices)
            - weighted_state_error_np(adv_rec, clean_inputs, state_indices=state_indices)
        )
        + action_weight * (((clean_act - adv_act) ** 2) - ((clean_act - adv_rec_act) ** 2))
    ).astype(np.float32)
    clean_benefit = (
        state_weight * (0.0 - weighted_state_error_np(clean_rec, clean_inputs, state_indices=state_indices))
        + action_weight * (0.0 - ((clean_act - clean_rec_act) ** 2))
    ).astype(np.float32)

    obs_inputs = np.concatenate([adv_inputs, clean_inputs], axis=0)
    rec_inputs = np.concatenate([adv_rec, clean_rec], axis=0)
    clean_refs = np.concatenate([clean_inputs, clean_inputs], axis=0)
    prev_obs_inputs = np.concatenate([adv_prev, clean_prev], axis=0)
    benefit_scores = np.concatenate([adv_benefit, clean_benefit], axis=0).astype(np.float32)
    attack_mask_full = np.concatenate(
        [
            attack_mask.astype(np.int64),
            np.zeros((clean_inputs.shape[0],), dtype=np.int64),
        ],
        axis=0,
    )
    time_indices_full = np.concatenate([time_indices, time_indices], axis=0)
    stations_full = np.concatenate([stations, stations], axis=0)
    is_new_arrivals_full = np.concatenate([is_new_arrivals, is_new_arrivals], axis=0)
    vehicle_ids_full = np.concatenate([vehicle_ids, vehicle_ids], axis=0)
    episode_indices_full = np.concatenate([episode_indices, episode_indices], axis=0)
    if label_mode == 'benefit':
        labels = (benefit_scores > margin).astype(np.int64)
    elif label_mode == 'attack':
        labels = attack_mask_full.astype(np.int64)
    else:
        raise ValueError(f'Unsupported posterior label mode: {posterior_label_mode!r}')

    keep_mask = (
        np.ones((labels.shape[0],), dtype=bool)
        if label_mode == 'attack' or margin <= 0.0
        else np.abs(benefit_scores) > margin
    )
    if not bool(np.any(keep_mask)):
        raise ValueError('Posterior detector dataset is empty after applying posterior benefit margin.')
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
    if label_mode == 'benefit' and bool(use_benefit_sample_weights):
        scale = max(float(np.quantile(np.abs(benefit_scores), 0.75)) if benefit_scores.size > 0 else 0.0, 1e-6)
        sample_weights = np.clip(np.abs(benefit_scores) / scale, 0.25, 4.0).astype(np.float32)
        sample_weights = sample_weights / max(float(np.mean(sample_weights)), 1e-6)

    metadata = {
        'collection_mode': f'unified_posterior_detector_{label_mode}_label',
        'detector_mode': 'posterior',
        'posterior_label_mode': label_mode,
        'profile_tag': str(profile_tag),
        'train_attacks': list(train_attack_tags),
        'samples': int(obs_inputs.shape[0]),
        'source_samples': int(clean_inputs.shape[0]),
        'attacked_samples': int(np.sum(attack_mask_full)),
        'clean_identity_samples': int(np.sum(attack_mask_full == 0)),
        'ambiguous_dropped_samples': int(np.sum(~keep_mask)),
        'positive_samples': int(np.sum(labels == 1)),
        'negative_samples': int(np.sum(labels == 0)),
        'benefit_margin': float(margin),
        'benefit_action_weight': float(action_weight),
        'benefit_state_weight': float(state_weight),
        'use_benefit_sample_weights': bool(label_mode == 'benefit' and bool(use_benefit_sample_weights)),
        'benefit_score_mean': float(np.mean(benefit_scores)) if benefit_scores.size else 0.0,
        'benefit_score_std': float(np.std(benefit_scores)) if benefit_scores.size else 0.0,
        'label_positive_rate': float(np.mean(labels == 1)) if labels.size else 0.0,
        'benefit_positive_rate': float(np.mean(benefit_scores > margin)) if benefit_scores.size else 0.0,
        'policy_input_mode': 'posterior_accept_recovered',
        'attack_trigger_mode': f'candidate_all_{state_scope}_obs',
        'state_scope': state_scope,
        'attack_state_scope': state_scope,
        'defense_state_scope': state_scope,
        'repair_mode': repair_mode,
        'posterior_candidate_state': 'core_only_injected' if repair_mode == 'core_only' else 'full_reconstruction',
        'attack_state_indices': list(state_indices),
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


def summarize_posterior_route_decisions(
    dataset: DetectorDatasetBundle,
    detector_model,
    detector_threshold: float,
    actor,
    device: torch.device,
) -> dict[str, float | int | str]:
    if dataset.obs_inputs is None or dataset.rec_inputs is None:
        raise ValueError('Posterior route summary requires obs_inputs and rec_inputs.')
    scores = posterior_detector_probabilities(
        detector_model,
        dataset.obs_inputs,
        dataset.rec_inputs,
        actor,
        device,
        time_indices=dataset.time_indices,
        stations=dataset.stations,
        is_new_arrivals=dataset.is_new_arrivals,
        prev_obs_inputs=dataset.prev_obs_inputs,
        include_temporal=bool(getattr(detector_model, 'include_temporal', True)),
    )
    routed = np.asarray(scores, dtype=np.float32).reshape(-1) >= float(detector_threshold)
    total = int(routed.shape[0])
    attack_mask = np.asarray(
        dataset.attack_mask if dataset.attack_mask is not None else np.zeros((total,), dtype=np.int64),
        dtype=np.int64,
    ).reshape(-1) > 0
    labels = None if dataset.labels is None else np.asarray(dataset.labels, dtype=np.int64).reshape(-1) > 0
    benefit_scores = None if dataset.benefit_scores is None else np.asarray(dataset.benefit_scores, dtype=np.float32).reshape(-1)
    margin = float((dataset.metadata or {}).get('benefit_margin', 0.0))
    benefit_positive = None if benefit_scores is None else benefit_scores > margin

    def count(mask: np.ndarray) -> int:
        return int(np.sum(np.asarray(mask, dtype=bool)))

    def rate(num: int, den: int) -> float:
        return 0.0 if int(den) <= 0 else float(num) / float(den)

    attack_count = count(attack_mask)
    clean_count = int(total - attack_count)
    attack_routed = count(attack_mask & routed)
    attack_rejected = count(attack_mask & ~routed)
    clean_routed = count(~attack_mask & routed)
    route_count = count(routed)
    summary: dict[str, float | int | str] = {
        'posterior_label_mode': canonical_posterior_label_mode((dataset.metadata or {}).get('posterior_label_mode', 'benefit')),
        'threshold': float(detector_threshold),
        'sample_count': total,
        'route_count': route_count,
        'route_rate': rate(route_count, total),
        'attack_sample_count': attack_count,
        'attack_route_count': attack_routed,
        'attack_route_rate': rate(attack_routed, attack_count),
        'attack_reject_count': attack_rejected,
        'attack_reject_rate': rate(attack_rejected, attack_count),
        'clean_sample_count': clean_count,
        'clean_route_count': clean_routed,
        'clean_route_rate': rate(clean_routed, clean_count),
    }
    if labels is not None:
        label_tp = count(routed & labels)
        label_fp = count(routed & ~labels)
        label_fn = count(~routed & labels)
        precision = rate(label_tp, label_tp + label_fp)
        recall = rate(label_tp, label_tp + label_fn)
        summary.update(
            {
                'label_positive_count': count(labels),
                'label_negative_count': int(total - count(labels)),
                'label_precision_at_threshold': precision,
                'label_recall_at_threshold': recall,
                'label_f1_at_threshold': 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall)),
            }
        )
    if benefit_positive is not None:
        attack_helpful = attack_mask & benefit_positive
        attack_unhelpful = attack_mask & ~benefit_positive
        helpful_count = count(attack_helpful)
        unhelpful_count = count(attack_unhelpful)
        unhelpful_rejected = count(attack_unhelpful & ~routed)
        unhelpful_routed = count(attack_unhelpful & routed)
        helpful_rejected = count(attack_helpful & ~routed)
        helpful_routed = count(attack_helpful & routed)
        summary.update(
            {
                'attack_benefit_positive_count': helpful_count,
                'attack_benefit_negative_count': unhelpful_count,
                'attack_benefit_negative_reject_count': unhelpful_rejected,
                'attack_benefit_negative_reject_rate': rate(unhelpful_rejected, unhelpful_count),
                'harmful_route_count': unhelpful_routed,
                'harmful_route_rate': rate(unhelpful_routed, unhelpful_count),
                'missed_helpful_count': helpful_rejected,
                'missed_helpful_rate': rate(helpful_rejected, helpful_count),
                'helpful_route_count': helpful_routed,
                'helpful_route_rate': rate(helpful_routed, helpful_count),
            }
        )
    return summary


def train_dae_for_job(
    bundle,
    actor,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_state: float,
    lambda_identity: float,
    validator=None,
    val_every: int = 1,
    select_by: str = 'reward_recovery',
    seq_len: int = 8,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    num_layers: int = 1,
    decoder_hidden_dim: int = 128,
    beta_kl: float = 1e-3,
    lambda_robust: float = 0.0,
    include_clean_sequences: bool = True,
    state_scope: str = 'local',
):
    return train_dae_from_bundle(
        bundle,
        actor,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_state=lambda_state,
        lambda_identity=lambda_identity,
        validator=validator,
        val_every=val_every,
        select_by=select_by,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        beta_kl=beta_kl,
        lambda_robust=lambda_robust,
        include_clean_sequences=include_clean_sequences,
        state_scope=state_scope,
    )


def _fresh_attacker_instance(attacker):
    if attacker is None:
        return None
    if hasattr(attacker, 'clone'):
        return attacker.clone()
    return attacker


def _make_dae_rollout_checkpoint_validator(
    args,
    arrivals: pd.DataFrame,
    actor,
    device: torch.device,
    *,
    train_jobs: list[dict],
    reward_profile: str,
    clean_penalty: float,
):
    reward = PROFILE_MAP[reward_profile]
    clean_baseline = rollout_episode(
        arrivals,
        actor,
        args.signals_path,
        device,
        reward,
        False,
        'O',
        None,
        None,
        None,
        'none',
        None,
        args.exploration_noise,
        args.price_threshold,
        args.soc_new_threshold,
        args.soc_rollout_threshold,
        args.even_station_target,
        args.odd_station_target,
        1.0,
        'obs',
        'posterior',
    )
    obs_low, obs_high = resolve_attack_obs_bounds(arrivals, args.signals_path)
    prepared_jobs: list[dict] = []
    for job in train_jobs:
        critic = load_attack_critic(job['algorithm'], device, bundle_path=resolve_attack_bundle_path(args))
        attacker = build_attacker(
            actor,
            device,
            algorithm=job['algorithm'],
            epsilon=float(job['epsilon']),
            alpha=args.alpha,
            iters=args.iters,
            seed=args.seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic,
            attack_state_scope=getattr(args, 'state_scope', 'local'),
        )
        scenario = canonical_attack_scenario(job['algorithm'], job['scenario'])
        attack_baseline = rollout_episode(
            arrivals,
            actor,
            args.signals_path,
            device,
            reward,
            True,
            scenario,
            _fresh_attacker_instance(attacker),
            None,
            None,
            'none',
            None,
            args.exploration_noise,
            args.price_threshold,
            args.soc_new_threshold,
            args.soc_rollout_threshold,
            args.even_station_target,
            args.odd_station_target,
            1.0,
            'obs',
            'posterior',
        )
        prepared_jobs.append({'job': job, 'scenario': scenario, 'attacker': attacker, 'attack_baseline': attack_baseline})

    clean_reward = float(clean_baseline['ep_reward'])

    def _validator(model):
        clean_dae = rollout_episode(
            arrivals,
            actor,
            args.signals_path,
            device,
            reward,
            False,
            'O',
            None,
            model,
            None,
            'always_dae',
            None,
            args.exploration_noise,
            args.price_threshold,
            args.soc_new_threshold,
            args.soc_rollout_threshold,
            args.even_station_target,
            args.odd_station_target,
            1.0,
            'obs',
            'posterior',
        )
        clean_dae_reward = float(clean_dae['ep_reward'])
        clean_drop_ratio = max(0.0, clean_reward - clean_dae_reward) / max(abs(clean_reward), 1e-6)
        recovery_ratios: list[float] = []
        attack_rewards: list[float] = []
        for item in prepared_jobs:
            attack_dae = rollout_episode(
                arrivals,
                actor,
                args.signals_path,
                device,
                reward,
                True,
                item['scenario'],
                _fresh_attacker_instance(item['attacker']),
                model,
                None,
                'always_dae',
                None,
                args.exploration_noise,
                args.price_threshold,
                args.soc_new_threshold,
                args.soc_rollout_threshold,
                args.even_station_target,
                args.odd_station_target,
                1.0,
                'obs',
                'posterior',
            )
            attack_reward = float(attack_dae['ep_reward'])
            attack_base_reward = float(item['attack_baseline']['ep_reward'])
            drop = clean_reward - attack_base_reward
            recovery_ratios.append(0.0 if abs(drop) < 1e-9 else float((attack_reward - attack_base_reward) / drop))
            attack_rewards.append(attack_reward)
        avg_recovery_ratio = float(np.mean(recovery_ratios)) if recovery_ratios else 0.0
        avg_attack_reward = float(np.mean(attack_rewards)) if attack_rewards else 0.0
        checkpoint_score = avg_recovery_ratio - float(clean_penalty) * clean_drop_ratio
        return {
            'dae_checkpoint_score': float(checkpoint_score),
            'rollout_recovery_ratio': avg_recovery_ratio,
            'rollout_avg_attack_dae_reward': avg_attack_reward,
            'rollout_clean_baseline_reward': clean_reward,
            'rollout_clean_dae_reward': clean_dae_reward,
            'rollout_clean_drop_ratio': float(clean_drop_ratio),
        }

    return _validator


def command_baseline_train(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    agent, history = train_agent(
        arrivals,
        args.signals_path,
        device,
        seed=args.seed,
        episodes=args.episodes,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        exploration_noise=args.exploration_noise,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        print_every=args.print_every,
        init_actor_path=args.init_actor_path,
        resume_bundle_path=args.resume_bundle_path,
        freeze_actor=args.freeze_actor,
        reward_profile=PROFILE_MAP[args.reward_profile],
    )
    model_path = resolve_baseline_actor_save_path(
        args.output_dir,
        args.actor_model_name,
        args.episodes,
        args.seed,
        args.reward_profile,
    )
    save_actor(agent.actor, model_path)
    bundle_path = args.bundle_path or resolve_baseline_bundle_save_path(
        args.output_dir,
        args.bundle_name,
        args.episodes,
        args.seed,
        args.reward_profile,
    )
    save_baseline_bundle(
        agent,
        bundle_path,
        metadata={
            'algorithm': 'baseline_ddpg',
            'policy_tag': 'baseline_best' if args.publish_policy else 'baseline',
            'episodes': int(args.episodes),
            'seed': int(args.seed),
            'buffer_size': int(args.buffer_size),
            'batch_size': int(args.batch_size),
            'learning_starts': int(args.learning_starts),
            'exploration_noise': float(args.exploration_noise),
            'gamma': float(args.gamma),
            'tau': float(args.tau),
            'actor_lr': float(args.actor_lr),
            'critic_lr': float(args.critic_lr),
            'reward_profile': str(args.reward_profile),
            'freeze_actor': bool(args.freeze_actor),
            'init_actor_path': None if args.init_actor_path is None else str(args.init_actor_path),
            'resume_bundle_path': None if args.resume_bundle_path is None else str(args.resume_bundle_path),
        },
    )
    history_path = model_path.with_name(f'{model_path.stem}_history.csv')
    save_train_history(history, history_path)
    if args.publish_policy:
        publish_best_baseline_artifacts(model_path, bundle_path)
    print('baseline-train complete')
    print(f'actor:   {model_path}')
    print(f'bundle:  {bundle_path}')
    print(f'history: {history_path}')


def command_train_sa_ddpg(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    warmup_steps = args.sa_warmup_steps if args.sa_warmup_steps is not None else args.learning_starts
    resolved_anchor_actor_path = args.sa_anchor_actor_path
    if resolved_anchor_actor_path is None and float(args.sa_anchor_reg_weight) > 0.0:
        resolved_anchor_actor_path = Path(DEFAULT_BASELINE_ACTOR_PATH)
    resolved_validation_baseline_bundle_path = args.sa_validation_baseline_bundle_path
    if int(args.sa_validation_every) > 0 and resolved_validation_baseline_bundle_path is None:
        resolved_validation_baseline_bundle_path = resolve_default_baseline_bundle_path()
    resolved_resume_bundle_path = args.resume_bundle_path
    resolved_init_actor_path = args.init_actor_path
    if resolved_resume_bundle_path is None and resolved_init_actor_path is None and bool(args.sa_baseline_warmstart):
        candidate = args.sa_init_bundle_path or resolve_default_baseline_bundle_path()
        candidate = Path(candidate).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f'SA-DDPG baseline warm-start bundle not found: {candidate}')
        resolved_resume_bundle_path = candidate
    elif resolved_resume_bundle_path is None and args.sa_init_bundle_path is not None:
        resolved_resume_bundle_path = Path(args.sa_init_bundle_path).expanduser().resolve()
    agent, history = train_sa_ddpg_agent(
        arrivals,
        args.signals_path,
        device,
        seed=args.seed,
        episodes=args.episodes,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        exploration_noise=args.exploration_noise,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        print_every=args.print_every,
        init_actor_path=resolved_init_actor_path,
        resume_bundle_path=resolved_resume_bundle_path,
        reward_profile=PROFILE_MAP[args.reward_profile],
        sa_train_attacks=split_csv_strings(args.sa_train_attacks),
        sa_epsilon=args.sa_epsilon,
        sa_alpha=args.sa_alpha,
        sa_steps=args.sa_steps,
        sa_objective=args.sa_objective,
        sa_noise_std=args.sa_noise_std,
        sa_rollout_attack_prob_start=args.sa_rollout_attack_prob_start,
        sa_rollout_attack_prob=args.sa_rollout_attack_prob,
        sa_update_attack_prob_start=args.sa_update_attack_prob_start,
        sa_update_attack_prob=args.sa_update_attack_prob,
        sa_curriculum_steps=args.sa_curriculum_steps,
        sa_warmup_steps=warmup_steps,
        sa_soc_new_threshold=args.sa_soc_new_threshold,
        sa_soc_rollout_threshold=args.sa_soc_rollout_threshold,
        sa_state_scope=args.sa_state_scope,
        sa_actor_reg_weight=args.sa_actor_reg_weight,
        sa_mixed_update_attacks=args.sa_mixed_update_attacks,
        sa_anchor_actor_path=resolved_anchor_actor_path,
        sa_anchor_reg_weight=args.sa_anchor_reg_weight,
        sa_anchor_clean_weight=args.sa_anchor_clean_weight,
        sa_clean_policy_weight=args.sa_clean_policy_weight,
        sa_risk_weight_scale=args.sa_risk_weight_scale,
        sa_risk_weight_max=args.sa_risk_weight_max,
        sa_risk_target_soc=args.sa_risk_target_soc,
        sa_validation_every=args.sa_validation_every,
        sa_validation_attacks=split_csv_strings(args.sa_validation_attacks) if args.sa_validation_attacks else None,
        sa_validation_baseline_bundle_path=resolved_validation_baseline_bundle_path,
        sa_validation_clean_drop_weight=args.sa_validation_clean_drop_weight,
        sa_validation_clean_drop_budget=args.sa_validation_clean_drop_budget,
        sa_validation_clean_drop_hard_cap=args.sa_validation_clean_drop_hard_cap,
        sa_validation_clean_exit_weight=args.sa_validation_clean_exit_weight,
        checkpoint_every=getattr(args, 'checkpoint_every', 0),
        checkpoint_dir=getattr(args, 'checkpoint_dir', None),
        checkpoint_prefix=getattr(args, 'checkpoint_prefix', 'sa_ddpg'),
        checkpoint_metadata={
            'algorithm': 'sa_ddpg',
            'policy_tag': 'sa_ddpg',
            'seed': int(args.seed),
            'episodes': int(args.episodes),
            'reward_profile': str(args.reward_profile),
            'sa_epsilon': float(args.sa_epsilon),
            'sa_train_attacks': split_csv_strings(args.sa_train_attacks),
            'sa_state_scope': str(args.sa_state_scope),
            'sa_baseline_warmstart': bool(args.sa_baseline_warmstart),
            'sa_init_bundle_path': None if args.sa_init_bundle_path is None else str(args.sa_init_bundle_path),
        },
    )
    model_path = resolve_sa_ddpg_actor_save_path(args.output_dir, args.actor_model_name, args.episodes, args.seed)
    save_actor(agent.actor, model_path)
    history_path = model_path.with_name(f'{model_path.stem}_history.csv')
    save_train_history(history, history_path)
    validation_history_path = None
    if getattr(history, 'validation_rows', None):
        validation_history_path = model_path.with_name(f'{model_path.stem}_validation.csv')
        normalize_result_frame(pd.DataFrame(history.validation_rows), rename_keys=False).to_csv(
            validation_history_path, index=False, float_format='%.4f'
        )
    test_history_path = None
    if getattr(history, 'test_rows', None):
        test_history_path = model_path.with_name(f'{model_path.stem}_test.csv')
        normalize_result_frame(pd.DataFrame(history.test_rows), rename_keys=False).to_csv(
            test_history_path, index=False, float_format='%.4f'
        )
    bundle_path = args.bundle_path or model_path.with_name(f'{model_path.stem}_bundle.pt')
    save_sa_ddpg_bundle(
        agent,
        bundle_path,
        metadata={
            'algorithm': 'sa_ddpg',
            'policy_tag': 'sa_ddpg',
            'episodes': int(args.episodes),
            'seed': int(args.seed),
            'buffer_size': int(args.buffer_size),
            'batch_size': int(args.batch_size),
            'learning_starts': int(args.learning_starts),
            'exploration_noise': float(args.exploration_noise),
            'gamma': float(args.gamma),
            'tau': float(args.tau),
            'actor_lr': float(args.actor_lr),
            'critic_lr': float(args.critic_lr),
            'reward_profile': str(args.reward_profile),
            'sa_epsilon': float(args.sa_epsilon),
            'sa_alpha': None if args.sa_alpha is None else float(args.sa_alpha),
            'sa_steps': None if args.sa_steps is None else int(args.sa_steps),
            'sa_objective': str(args.sa_objective),
            'sa_noise_std': float(args.sa_noise_std),
            'sa_train_attacks': split_csv_strings(args.sa_train_attacks),
            'sa_state_scope': str(args.sa_state_scope),
            'sa_actor_reg_weight': float(args.sa_actor_reg_weight),
            'sa_mixed_update_attacks': bool(args.sa_mixed_update_attacks),
            'sa_anchor_actor_path': None if resolved_anchor_actor_path is None else str(resolved_anchor_actor_path),
            'sa_anchor_reg_weight': float(args.sa_anchor_reg_weight),
            'sa_anchor_clean_weight': float(args.sa_anchor_clean_weight),
            'sa_clean_policy_weight': float(args.sa_clean_policy_weight),
            'sa_risk_weight_scale': float(args.sa_risk_weight_scale),
            'sa_risk_weight_max': float(args.sa_risk_weight_max),
            'sa_risk_target_soc': None if args.sa_risk_target_soc is None else float(args.sa_risk_target_soc),
            'sa_validation_every': int(args.sa_validation_every),
            'sa_validation_attacks': split_csv_strings(args.sa_validation_attacks) if args.sa_validation_attacks else split_csv_strings(args.sa_train_attacks),
            'sa_validation_baseline_bundle_path': None if resolved_validation_baseline_bundle_path is None else str(resolved_validation_baseline_bundle_path),
            'sa_validation_clean_drop_weight': float(args.sa_validation_clean_drop_weight),
            'sa_validation_clean_drop_budget': float(args.sa_validation_clean_drop_budget),
            'sa_validation_clean_drop_hard_cap': float(args.sa_validation_clean_drop_hard_cap),
            'sa_validation_clean_exit_weight': float(args.sa_validation_clean_exit_weight),
            'sa_best_validation': getattr(agent, 'best_validation', None),
            'sa_rollout_attack_prob_start': float(args.sa_rollout_attack_prob_start),
            'sa_rollout_attack_prob': float(args.sa_rollout_attack_prob),
            'sa_update_attack_prob_start': float(args.sa_update_attack_prob_start),
            'sa_update_attack_prob': float(args.sa_update_attack_prob),
            'sa_curriculum_steps': int(args.sa_curriculum_steps),
            'sa_warmup_steps': int(warmup_steps),
            'sa_soc_new_threshold': float(args.sa_soc_new_threshold),
            'sa_soc_rollout_threshold': float(args.sa_soc_rollout_threshold),
            'sa_baseline_warmstart': bool(args.sa_baseline_warmstart),
            'sa_init_bundle_path': None if args.sa_init_bundle_path is None else str(args.sa_init_bundle_path),
            'init_actor_path': None if resolved_init_actor_path is None else str(resolved_init_actor_path),
            'resume_bundle_path': None if resolved_resume_bundle_path is None else str(resolved_resume_bundle_path),
        },
    )
    print('train-sa-ddpg complete')
    print(f'actor:   {model_path}')
    print(f'bundle:  {bundle_path}')
    print(f'history: {history_path}')
    if validation_history_path is not None:
        print(f'validation: {validation_history_path}')




def command_train_atla_ddpg(args):
    """Train ATLA-DDPG with the PPO-ATLA learned observation-adversary mechanism.

    Only the agent optimizer/policy family is changed from PPO-LSTM to DDPG.
    The training adversary is a learned MLP observation adversary, not a rotating
    PGD/Q/FGSM attack list.  PGD/Q/FGSM remain evaluation attacks only.
    """
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)

    resolved_validation_baseline_bundle_path = args.sa_validation_baseline_bundle_path
    if int(args.sa_validation_every) > 0 and resolved_validation_baseline_bundle_path is None:
        resolved_validation_baseline_bundle_path = resolve_default_baseline_bundle_path(args.reward_profile)

    resolved_init_bundle_path = None
    resolved_init_actor_path = args.init_actor_path
    if bool(args.sa_baseline_warmstart):
        candidate = args.sa_init_bundle_path or resolve_default_baseline_bundle_path(args.reward_profile)
        candidate = Path(candidate).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f'ATLA-DDPG baseline warm-start bundle not found: {candidate}')
        resolved_init_bundle_path = candidate
        resolved_init_actor_path = None
    elif args.sa_init_bundle_path is not None:
        resolved_init_bundle_path = Path(args.sa_init_bundle_path).expanduser().resolve()
        resolved_init_actor_path = None

    resolved_anchor_actor_path = args.sa_anchor_actor_path
    if resolved_anchor_actor_path is None and float(args.sa_anchor_reg_weight) > 0.0 and resolved_init_bundle_path is None:
        resolved_anchor_actor_path = Path(DEFAULT_BASELINE_ACTOR_PATH)

    checkpoint_metadata = {
        'algorithm': 'atla_ddpg',
        'policy_tag': 'atla_ddpg',
        'source_variant': 'learned_adversary_ddpg_agent',
        'source_algorithm': 'atla_ddpg_learned_adversary',
        'ppo_line_touched': False,
        'training_adversary_type': 'learned_mlp_observation_adversary',
        'manual_attack_rotation_used_for_training': False,
        'train_attack_list_used': False,
        'note': 'DDPG replacement of PPO-ATLA agent while preserving learned observation adversary and alternating agent/adversary phases. External PGD/Q/FGSM are evaluation only.',
        'seed': int(args.seed),
        'episodes': int(args.episodes),
        'buffer_size': int(args.buffer_size),
        'batch_size': int(args.batch_size),
        'learning_starts': int(args.learning_starts),
        'exploration_noise': float(args.exploration_noise),
        'gamma': float(args.gamma),
        'tau': float(args.tau),
        'actor_lr': float(args.actor_lr),
        'critic_lr': float(args.critic_lr),
        'adv_actor_lr': float(args.adv_actor_lr),
        'adv_critic_lr': float(args.adv_critic_lr),
        'reward_profile': str(args.reward_profile),
        'sa_epsilon': float(args.sa_epsilon),
        'epsilon': float(args.sa_epsilon),
        'sa_state_scope': str(args.sa_state_scope),
        'atla_state_scope': str(args.sa_state_scope),
        'state_scope': str(args.sa_state_scope),
        'train_scope': str(args.sa_state_scope),
        'sa_actor_reg_weight': float(args.sa_actor_reg_weight),
        'sa_anchor_actor_path': None if resolved_anchor_actor_path is None else str(resolved_anchor_actor_path),
        'sa_anchor_reg_weight': float(args.sa_anchor_reg_weight),
        'sa_anchor_clean_weight': float(args.sa_anchor_clean_weight),
        'sa_validation_every': int(args.sa_validation_every),
        'sa_validation_baseline_bundle_path': None if resolved_validation_baseline_bundle_path is None else str(resolved_validation_baseline_bundle_path),
        'sa_baseline_warmstart': bool(args.sa_baseline_warmstart),
        'sa_init_bundle_path': None if resolved_init_bundle_path is None else str(resolved_init_bundle_path),
        'init_actor_path': None if resolved_init_actor_path is None else str(resolved_init_actor_path),
    }

    agent, history = train_atla_ddpg_learned_adversary_agent(
        arrivals,
        args.signals_path,
        device,
        seed=args.seed,
        episodes=args.episodes,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        exploration_noise=args.exploration_noise,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        adv_actor_lr=args.adv_actor_lr,
        adv_critic_lr=args.adv_critic_lr,
        ppo_epochs=args.ppo_epochs,
        num_minibatches=args.num_minibatches,
        adv_entropy_coeff=args.adv_entropy_coeff,
        print_every=args.print_every,
        init_bundle_path=resolved_init_bundle_path,
        init_actor_path=resolved_init_actor_path,
        anchor_actor_path=resolved_anchor_actor_path,
        reward_profile=PROFILE_MAP[args.reward_profile],
        epsilon=args.sa_epsilon,
        attack_state_scope=args.sa_state_scope,
        actor_reg_weight=args.sa_actor_reg_weight,
        anchor_reg_weight=args.sa_anchor_reg_weight,
        anchor_clean_weight=args.sa_anchor_clean_weight,
        validation_every=args.sa_validation_every,
        validation_baseline_bundle_path=resolved_validation_baseline_bundle_path,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        checkpoint_metadata=checkpoint_metadata,
    )
    model_path = resolve_atla_ddpg_actor_save_path(args.output_dir, args.actor_model_name, args.episodes, args.seed, args.sa_state_scope)
    save_actor(agent.actor, model_path)
    history_path = args.history_path or model_path.with_name(f'{model_path.stem}_history.csv')
    save_train_history(history, history_path)
    validation_history_path = None
    if getattr(history, 'validation_rows', None):
        validation_history_path = model_path.with_name(f'{model_path.stem}_validation.csv')
        normalize_result_frame(pd.DataFrame(history.validation_rows), rename_keys=False).to_csv(
            validation_history_path, index=False, float_format='%.4f'
        )
    bundle_path = args.bundle_path or model_path.with_name(f'{model_path.stem}_bundle.pt')
    metadata = {**checkpoint_metadata, 'atla_ddpg_best_validation': getattr(agent, 'best_validation', None)}
    save_atla_ddpg_learned_bundle(agent, bundle_path, metadata=metadata)
    print('train-atla-ddpg complete')
    print(f'actor:   {model_path}')
    print(f'bundle:  {bundle_path}')
    print(f'history: {history_path}')
    if validation_history_path is not None:
        print(f'validation: {validation_history_path}')

def command_train_online_ppo_lstm(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    agent, history = train_online_ppo_lstm_agent(
        arrivals,
        args.signals_path,
        device,
        seed=args.seed,
        outer_iters=args.outer_iters,
        phase_steps=args.phase_steps,
        ppo_epochs=args.ppo_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coeff=args.entropy_coeff,
        reward_profile=PROFILE_MAP[args.reward_profile],
        hidden_dim=args.hidden_dim,
        lstm_dim=args.lstm_dim,
        print_every=args.print_every,
        validation_every=args.validation_every,
        init_bundle_path=args.init_bundle_path,
        iteration_offset=args.iteration_offset,
    )
    output_dir = ensure_dir(args.output_dir)
    bundle_path = args.bundle_path or (output_dir / 'default_bundle.pt')
    history_path = args.history_path or Path(bundle_path).with_name(f'{Path(bundle_path).stem}_history.csv')
    save_online_ppo_lstm_bundle(
        agent,
        bundle_path,
        metadata={
            'algorithm': 'online_ppo_lstm',
            'policy_tag': 'online_ppo_lstm',
            'online_atla_algorithm_variant': 'ppo_lstm_clean',
            'seed': int(args.seed),
            'outer_iters': int(args.outer_iters),
            'phase_steps': int(args.phase_steps),
            'ppo_epochs': int(args.ppo_epochs),
            'num_minibatches': int(args.num_minibatches),
            'gamma': float(args.gamma),
            'gae_lambda': float(args.gae_lambda),
            'clip_eps': float(args.clip_eps),
            'actor_lr': float(args.actor_lr),
            'critic_lr': float(args.critic_lr),
            'entropy_coeff': float(args.entropy_coeff),
            'epsilon': 0.0,
            'attack_state_scope': 'none',
            'sa_reg_weight': 0.0,
            'sa_reg_steps': 0,
            'reward_profile': str(args.reward_profile),
            'hidden_dim': int(args.hidden_dim),
            'lstm_dim': int(args.lstm_dim),
            'adversary_hidden_dim': 128,
            'data_path': str(args.data_path),
            'signals_path': str(args.signals_path),
            'max_sessions': None if args.max_sessions is None else int(args.max_sessions),
        },
    )
    save_online_ppo_lstm_history(history, history_path)
    print('train-online-ppo-lstm complete')
    print(f'bundle:  {bundle_path}')
    print(f'history: {history_path}')


def command_eval_online_ppo_lstm(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    scopes = split_attack_state_scopes(args.attack_state_scope)
    bundle_path = args.bundle_path or DEFAULT_ONLINE_PPO_LSTM_BUNDLE
    if not Path(bundle_path).exists():
        raise FileNotFoundError(
            f'Default PPO-LSTM bundle not found: {bundle_path}. '
            f'Train it with train-online-ppo-lstm or pass --bundle-path explicitly.'
        )
    agent = load_online_ppo_lstm_bundle(bundle_path, device)
    df, attack_histories = evaluate_online_ppo_lstm_agent(
        arrivals,
        args.signals_path,
        agent,
        reward_profile=PROFILE_MAP[args.reward_profile],
        epsilon=args.epsilon,
        attack_state_scopes=scopes,
        seed=args.seed,
        eval_adv_iters=args.eval_adv_iters,
        eval_adv_phase_steps=args.eval_adv_phase_steps,
        ppo_epochs=args.ppo_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        adv_actor_lr=args.adv_actor_lr,
        adv_critic_lr=args.adv_critic_lr,
        adv_entropy_coeff=args.adv_entropy_coeff,
    )
    df.insert(0, 'model_scope', 'ppo_lstm')
    df.insert(1, 'bundle_path', str(bundle_path))
    output_dir = ensure_dir(args.output_dir)
    bundle_stem = Path(bundle_path).stem if args.bundle_path is not None else 'default'
    eval_path = args.eval_path or (output_dir / f'{bundle_stem}_eval.csv')
    save_online_ppo_lstm_eval(df, eval_path)
    for scope, rows in attack_histories.items():
        if rows:
            hist_path = Path(eval_path).with_name(f'{Path(eval_path).stem}_{scope}_learned_attack_history.csv')
            normalize_result_frame(pd.DataFrame(rows), rename_keys=False).to_csv(hist_path, index=False, float_format='%.4f')
    print('eval-online-ppo-lstm complete')
    print(df[[col for col in ['model_scope', 'scope', 'attack_mode', 'ep_reward', 'Natural Reward', 'Worst Attack Reward', 'Robust Ratio'] if col in df.columns]].to_string(index=False))
    print(f'eval: {eval_path}')



def command_train_online_atla_ppo_lstm_sa(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    agent, history = train_online_atla_ppo_lstm_sa_agent(
        arrivals,
        args.signals_path,
        device,
        seed=args.seed,
        outer_iters=args.outer_iters,
        phase_steps=args.phase_steps,
        ppo_epochs=args.ppo_epochs,
        num_minibatches=args.num_minibatches,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        adv_actor_lr=args.adv_actor_lr,
        adv_critic_lr=args.adv_critic_lr,
        entropy_coeff=args.entropy_coeff,
        adv_entropy_coeff=args.adv_entropy_coeff,
        sa_reg_weight=args.sa_reg_weight,
        sa_reg_steps=args.sa_reg_steps,
        epsilon=args.epsilon,
        attack_state_scope=args.attack_state_scope,
        reward_profile=PROFILE_MAP[args.reward_profile],
        hidden_dim=args.hidden_dim,
        lstm_dim=args.lstm_dim,
        adversary_hidden_dim=args.adversary_hidden_dim,
        print_every=args.print_every,
        validation_every=args.validation_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        init_bundle_path=args.init_bundle_path,
        iteration_offset=args.iteration_offset,
    )
    output_dir = ensure_dir(args.output_dir)
    tag = 'atla'
    bundle_path = args.bundle_path or (output_dir / 'atla_bundle.pt')
    history_path = args.history_path or Path(bundle_path).with_name(f'{Path(bundle_path).stem}_history.csv')
    save_online_atla_ppo_lstm_sa_bundle(
        agent,
        bundle_path,
        metadata={
            'algorithm': 'atla',
            'policy_tag': 'atla',
            'online_atla_algorithm_variant': 'random_init_partial_sa_reg',
            'seed': int(args.seed),
            'outer_iters': int(args.outer_iters),
            'phase_steps': int(args.phase_steps),
            'ppo_epochs': int(args.ppo_epochs),
            'num_minibatches': int(args.num_minibatches),
            'gamma': float(args.gamma),
            'gae_lambda': float(args.gae_lambda),
            'clip_eps': float(args.clip_eps),
            'actor_lr': float(args.actor_lr),
            'critic_lr': float(args.critic_lr),
            'adv_actor_lr': float(args.adv_actor_lr),
            'adv_critic_lr': float(args.adv_critic_lr),
            'entropy_coeff': float(args.entropy_coeff),
            'adv_entropy_coeff': float(args.adv_entropy_coeff),
            'sa_reg_weight': float(args.sa_reg_weight),
            'sa_reg_steps': int(args.sa_reg_steps),
            'epsilon': float(args.epsilon),
            'attack_state_scope': str(args.attack_state_scope),
            'reward_profile': str(args.reward_profile),
            'hidden_dim': int(args.hidden_dim),
            'lstm_dim': int(args.lstm_dim),
            'adversary_hidden_dim': int(args.adversary_hidden_dim),
            'data_path': str(args.data_path),
            'signals_path': str(args.signals_path),
            'max_sessions': None if args.max_sessions is None else int(args.max_sessions),
        },
    )
    save_atla_ppo_lstm_sa_history(history, history_path)
    print('train-atla complete')
    print(f'bundle:  {bundle_path}')
    print(f'history: {history_path}')


def command_eval_online_atla_ppo_lstm_sa(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    scopes = split_attack_state_scopes(args.attack_state_scope)
    default_bundles = {
        'local': DEFAULT_ONLINE_ATLA_PPO_LSTM_SA_LOCAL_BUNDLE,
        'all': DEFAULT_ONLINE_ATLA_PPO_LSTM_SA_ALL_BUNDLE,
    }
    eval_frames: list[pd.DataFrame] = []
    attack_histories: dict[str, list[dict]] = {}
    if args.bundle_path is not None:
        agent = load_online_atla_ppo_lstm_sa_bundle(args.bundle_path, device)
        df, attack_histories = evaluate_online_atla_ppo_lstm_sa_agent(
            arrivals,
            args.signals_path,
            agent,
            reward_profile=PROFILE_MAP[args.reward_profile],
            epsilon=args.epsilon,
            attack_state_scopes=scopes,
            seed=args.seed,
            eval_adv_iters=args.eval_adv_iters,
            eval_adv_phase_steps=args.eval_adv_phase_steps,
            ppo_epochs=args.ppo_epochs,
            num_minibatches=args.num_minibatches,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            adv_actor_lr=args.adv_actor_lr,
            adv_critic_lr=args.adv_critic_lr,
            adv_entropy_coeff=args.adv_entropy_coeff,
        )
        eval_frames.append(df)
        bundle_stem = Path(args.bundle_path).stem
    else:
        for scope in scopes:
            canonical_scope = str(scope)
            if canonical_scope not in default_bundles:
                raise ValueError(f'No default ATLA bundle is configured for scope: {canonical_scope}')
            bundle_path = default_bundles[canonical_scope]
            if not Path(bundle_path).exists():
                raise FileNotFoundError(
                    f'Default ATLA bundle not found: {bundle_path}. '
                    f'Train it with train-atla or pass --bundle-path explicitly.'
                )
            agent = load_online_atla_ppo_lstm_sa_bundle(bundle_path, device)
            df_scope, histories_scope = evaluate_online_atla_ppo_lstm_sa_agent(
                arrivals,
                args.signals_path,
                agent,
                reward_profile=PROFILE_MAP[args.reward_profile],
                epsilon=args.epsilon,
                attack_state_scopes=[canonical_scope],
                seed=args.seed,
                eval_adv_iters=args.eval_adv_iters,
                eval_adv_phase_steps=args.eval_adv_phase_steps,
                ppo_epochs=args.ppo_epochs,
                num_minibatches=args.num_minibatches,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_eps=args.clip_eps,
                adv_actor_lr=args.adv_actor_lr,
                adv_critic_lr=args.adv_critic_lr,
                adv_entropy_coeff=args.adv_entropy_coeff,
            )
            df_scope.insert(0, 'model_scope', canonical_scope)
            df_scope.insert(1, 'bundle_path', str(bundle_path))
            eval_frames.append(df_scope)
            attack_histories.update({history_scope: rows for history_scope, rows in histories_scope.items()})
        df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
        bundle_stem = 'atla'
    output_dir = ensure_dir(args.output_dir)
    eval_path = args.eval_path or (output_dir / f'{bundle_stem}_eval.csv')
    save_atla_ppo_lstm_sa_eval(df, eval_path)
    for scope, rows in attack_histories.items():
        if rows:
            hist_path = Path(eval_path).with_name(f'{Path(eval_path).stem}_{scope}_learned_attack_history.csv')
            normalize_result_frame(pd.DataFrame(rows), rename_keys=False).to_csv(hist_path, index=False, float_format='%.4f')
    print('eval-atla complete')
    print(df[[col for col in ['model_scope', 'scope', 'attack_mode', 'ep_reward', 'Natural Reward', 'Worst Attack Reward', 'Robust Ratio'] if col in df.columns]].to_string(index=False))
    print(f'eval: {eval_path}')


def _load_actor_and_critic_for_eval(
    *,
    label: str,
    actor_path: Path | None,
    bundle_path: Path | None,
    device: torch.device,
) -> tuple[Actor, Critic | None, Path, Path | None]:
    resolved_actor_path = Path(actor_path or bundle_path).expanduser().resolve() if (actor_path or bundle_path) else None
    if resolved_actor_path is None:
        raise ValueError(f'{label} requires an actor path or a bundle path.')
    actor = load_actor_from_path(resolved_actor_path, device)
    resolved_bundle_path = Path(bundle_path).expanduser().resolve() if bundle_path is not None else None
    if resolved_bundle_path is None and _torch_artifact_has_critic(resolved_actor_path):
        resolved_bundle_path = resolved_actor_path
    critic = None
    if resolved_bundle_path is not None:
        payload = load_actor_critic_bundle(resolved_bundle_path, device)
        critic_state = payload.get('critic_state_dict')
        if critic_state is not None:
            critic = Critic().to(device)
            critic.load_state_dict(critic_state)
            critic.eval()
    return actor, critic, resolved_actor_path, resolved_bundle_path


def _require_attack_critic(label: str, algorithm: str, critic: Critic | None, bundle_path: Path | None) -> Critic | None:
    if canonical_attack_algorithm(algorithm) != 'q_function':
        return None
    if critic is None:
        raise ValueError(f'{label} q_function evaluation requires a matching critic bundle; got {bundle_path}.')
    return critic


def _load_shared_attack_oracle_critic(bundle_path: Path | None, device: torch.device) -> Critic | None:
    if bundle_path is None:
        return None
    payload = torch.load(Path(bundle_path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        return None
    critic_state = payload.get('attack_oracle_critic_state_dict')
    if critic_state is None:
        return None
    critic = Critic().to(device)
    critic.load_state_dict(critic_state)
    critic.eval()
    return critic


def _shared_recovery_attack_critic(
    algorithm: str,
    *,
    sa_oracle_critic: Critic | None,
    baseline_critic: Critic | None,
    baseline_bundle_path: Path | None,
) -> Critic | None:
    if canonical_attack_algorithm(algorithm) != 'q_function':
        return None
    if sa_oracle_critic is not None:
        return sa_oracle_critic
    if baseline_critic is None:
        raise ValueError(
            'q_function recovery evaluation requires a baseline critic bundle '
            f'to serve as the shared attack critic; got {baseline_bundle_path}.'
        )
    return baseline_critic


def command_evaluate_sa_ddpg(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    exploration_noise = float(getattr(args, 'exploration_noise', 0.0))
    baseline_bundle_path = Path(args.baseline_bundle_path).expanduser().resolve() if args.baseline_bundle_path is not None else resolve_default_baseline_bundle_path()
    baseline_actor_path = Path(args.baseline_actor_path).expanduser().resolve() if args.baseline_actor_path is not None else Path(DEFAULT_BASELINE_ACTOR_PATH)
    baseline_actor, baseline_critic, resolved_baseline_actor_path, resolved_baseline_bundle_path = _load_actor_and_critic_for_eval(
        label='baseline policy',
        actor_path=baseline_actor_path,
        bundle_path=baseline_bundle_path,
        device=device,
    )
    sa_actor, sa_critic, resolved_sa_actor_path, resolved_sa_bundle_path = _load_actor_and_critic_for_eval(
        label='SA-DDPG policy',
        actor_path=args.sa_actor_path,
        bundle_path=args.sa_bundle_path,
        device=device,
    )
    shared_oracle_critic = _load_shared_attack_oracle_critic(resolved_sa_bundle_path, device)
    sa_metadata = _read_torch_artifact_metadata(resolved_sa_bundle_path) if resolved_sa_bundle_path is not None else {}
    eval_algorithms = split_csv_strings(args.eval_algorithms)
    online_algorithm = str(sa_metadata.get('algorithm', '')).strip().lower()
    retained_online_algorithms = {'atla', 'atla_ddpg'}
    allow_cross_attack_eval = bool(getattr(args, 'allow_cross_attack_eval', False))
    if online_algorithm in {'sa_ddpg', *retained_online_algorithms}:
        actual_scope_value = (
            sa_metadata.get('wocar_state_scope')
            or sa_metadata.get('owocar_state_scope')
            or sa_metadata.get('sa_state_scope')
            or sa_metadata.get('atla_state_scope')
            or sa_metadata.get('state_scope')
            or sa_metadata.get('train_scope')
        )
        if actual_scope_value is not None:
            actual_scope = canonical_state_scope(str(actual_scope_value))
            expected_scope = canonical_state_scope(args.state_scope)
            if actual_scope != expected_scope:
                raise ValueError(
                    f'{online_algorithm} evaluation state_scope mismatch: '
                    f'bundle {resolved_sa_bundle_path} was trained with {actual_scope!r}, '
                    f'but evaluation requested {expected_scope!r}. '
                    'Train/evaluate online robust checkpoints separately for all/local scopes.'
                )
        if online_algorithm not in {'atla'}:
            train_attacks_raw = (
                sa_metadata.get('wocar_train_attacks')
                or sa_metadata.get('owocar_train_attacks')
                or sa_metadata.get('sa_train_attacks')
                or sa_metadata.get('train_attacks')
            )
            if train_attacks_raw is not None:
                if isinstance(train_attacks_raw, str):
                    train_attacks = split_csv_strings(train_attacks_raw)
                else:
                    train_attacks = list(train_attacks_raw)
                train_attack_set = {canonical_sa_train_attack(item) for item in train_attacks if str(item).strip()}
                eval_attack_set = {canonical_sa_train_attack(item) for item in eval_algorithms if str(item).strip()}
                missing_attacks = sorted(eval_attack_set.difference(train_attack_set))
                if missing_attacks and not allow_cross_attack_eval:
                    raise ValueError(
                        f'{online_algorithm} evaluation attack mismatch: '
                        f'bundle {resolved_sa_bundle_path} was trained with {sorted(train_attack_set)!r}, '
                        f'but evaluation requested {sorted(eval_attack_set)!r}. '
                        'Use a matching checkpoint for the main table, pass --allow-cross-attack-eval, '
                        'or run cross-attack evaluation separately.'
                    )
                if missing_attacks and allow_cross_attack_eval:
                    print(
                        f'[evaluate-sa-ddpg] cross-attack eval enabled: bundle trained with '
                        f'{sorted(train_attack_set)!r}; evaluating extra attacks {missing_attacks!r}.'
                    )
    epsilons = split_csv_floats(args.epsilons)
    eval_scenarios = split_csv_strings(getattr(args, 'eval_scenarios', 'O'))
    jobs = list(iter_attack_jobs(eval_algorithms, eval_scenarios, epsilons))
    reward_profile = PROFILE_MAP[args.reward_profile]
    obs_low, obs_high = resolve_attack_obs_bounds(arrivals, args.signals_path)
    baseline_clean = rollout_episode(
        arrivals,
        baseline_actor,
        args.signals_path,
        device,
        reward_profile,
        False,
        'O',
        None,
        None,
        None,
        'none',
        None,
        exploration_noise,
        args.price_threshold,
        args.soc_new_threshold,
        args.soc_rollout_threshold,
        args.even_station_target,
        args.odd_station_target,
        args.attack_ratio,
        args.attack_scope,
        'posterior',
    )
    sa_clean = rollout_episode(
        arrivals,
        sa_actor,
        args.signals_path,
        device,
        reward_profile,
        False,
        'O',
        None,
        None,
        None,
        'none',
        None,
        exploration_noise,
        args.price_threshold,
        args.soc_new_threshold,
        args.soc_rollout_threshold,
        args.even_station_target,
        args.odd_station_target,
        1.0,
        'obs',
        'posterior',
    )

    rows: list[dict] = []
    manifest_rows: list[dict] = []
    for job in jobs:
        algorithm = str(job['algorithm'])
        epsilon = float(job['epsilon'])
        scenario = canonical_attack_scenario(algorithm, job['scenario'])
        shared_attack_critic = _shared_recovery_attack_critic(
            algorithm,
            sa_oracle_critic=shared_oracle_critic,
            baseline_critic=baseline_critic,
            baseline_bundle_path=resolved_baseline_bundle_path,
        )
        baseline_attacker = build_attacker(
            baseline_actor,
            device,
            algorithm=algorithm,
            epsilon=epsilon,
            alpha=args.alpha,
            iters=args.iters,
            seed=args.seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=shared_attack_critic,
            attack_state_scope=args.state_scope,
        )
        sa_attacker = build_attacker(
            sa_actor,
            device,
            algorithm=algorithm,
            epsilon=epsilon,
            alpha=args.alpha,
            iters=args.iters,
            seed=args.seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=shared_attack_critic,
            attack_state_scope=args.state_scope,
        )
        baseline_attack = rollout_episode(
            arrivals,
            baseline_actor,
            args.signals_path,
            device,
            reward_profile,
            True,
            scenario,
            baseline_attacker,
            None,
            None,
            'none',
            None,
            exploration_noise,
            args.price_threshold,
            args.soc_new_threshold,
            args.soc_rollout_threshold,
            args.even_station_target,
            args.odd_station_target,
            args.attack_ratio,
            args.attack_scope,
            'posterior',
        )
        sa_attack = rollout_episode(
            arrivals,
            sa_actor,
            args.signals_path,
            device,
            reward_profile,
            True,
            scenario,
            sa_attacker,
            None,
            None,
            'none',
            None,
            exploration_noise,
            args.price_threshold,
            args.soc_new_threshold,
            args.soc_rollout_threshold,
            args.even_station_target,
            args.odd_station_target,
            args.attack_ratio,
            args.attack_scope,
            'posterior',
        )
        baseline_clean_reward = float(baseline_clean['ep_reward'])
        baseline_attack_reward = float(baseline_attack['ep_reward'])
        sa_clean_reward = float(sa_clean['ep_reward'])
        sa_attack_reward = float(sa_attack['ep_reward'])
        attack_drop = baseline_clean_reward - baseline_attack_reward
        clean_drop = baseline_clean_reward - sa_clean_reward
        clean_drop_ratio = 0.0 if abs(baseline_clean_reward) < 1e-9 else clean_drop / abs(baseline_clean_reward)
        recovery_ratio = 0.0 if abs(attack_drop) < 1e-9 else (sa_attack_reward - baseline_attack_reward) / attack_drop
        rows.append(
            {
                'attack': str(job['attack_tag']),
                'epsilon': epsilon,
                'baseline_clean_reward': baseline_clean_reward,
                'baseline_attack_reward': baseline_attack_reward,
                'clean_reward': sa_clean_reward,
                'attack_reward': sa_attack_reward,
                'recovery_ratio': float(recovery_ratio),
                'clean_drop': float(clean_drop),
                'clean_drop_ratio': float(clean_drop_ratio),
                'attack_drop_vs_own_clean': float(sa_clean_reward - sa_attack_reward),
                'baseline_attack_drop_vs_clean': float(baseline_clean_reward - baseline_attack_reward),
                'baseline_clean_dense_safety': float(baseline_clean.get('ep_r4_dense', 0.0)),
                'baseline_attack_dense_safety': float(baseline_attack.get('ep_r4_dense', 0.0)),
                'clean_dense_safety': float(sa_clean.get('ep_r4_dense', 0.0)),
                'attack_dense_safety': float(sa_attack.get('ep_r4_dense', 0.0)),
                'baseline_exit_vio': int(baseline_clean['exit_vio']),
                'baseline_attack_exit_vio': int(baseline_attack['exit_vio']),
                'clean_exit_vio': int(sa_clean['exit_vio']),
                'attack_exit_vio': int(sa_attack['exit_vio']),
                'baseline_run_vio': int(baseline_clean.get('run_vio', 0)),
                'baseline_attack_run_vio': int(baseline_attack.get('run_vio', 0)),
                'clean_run_vio': int(sa_clean.get('run_vio', 0)),
                'attack_run_vio': int(sa_attack.get('run_vio', 0)),
                'baseline_attack_action_abs_diff_mean': float(baseline_attack.get('attack_action_abs_diff_mean', 0.0)),
                'attack_action_abs_diff_mean': float(sa_attack.get('attack_action_abs_diff_mean', 0.0)),
                'baseline_attack_delta_linf_mean': float(baseline_attack.get('attack_delta_linf_mean', 0.0)),
                'attack_delta_linf_mean': float(sa_attack.get('attack_delta_linf_mean', 0.0)),
            }
        )
        manifest_rows.append(
            {
                'attack': str(job['attack_tag']),
                'algorithm': algorithm,
                'epsilon': epsilon,
                'state_scope': str(args.state_scope),
                'attack_ratio': float(args.attack_ratio),
                'attack_scope': str(args.attack_scope),
                'q_function_attack_critic_source': (
                    'oracle' if canonical_attack_algorithm(algorithm) == 'q_function' and shared_oracle_critic is not None
                    else 'baseline' if canonical_attack_algorithm(algorithm) == 'q_function'
                    else 'n/a'
                ),
                'baseline_actor_path': str(resolved_baseline_actor_path),
                'baseline_bundle_path': None if resolved_baseline_bundle_path is None else str(resolved_baseline_bundle_path),
                'sa_actor_path': str(resolved_sa_actor_path),
                'sa_bundle_path': None if resolved_sa_bundle_path is None else str(resolved_sa_bundle_path),
            }
        )

    out_dir = ensure_dir(Path(args.save_dir or (DEFAULT_RESULTS_DIR / 'sa_ddpg')) / f"eval_{Path(resolved_sa_actor_path).stem}_{str(args.state_scope)}")
    df = normalize_result_frame(pd.DataFrame(rows), rename_keys=False, digits=4)
    manifest_df = normalize_result_frame(pd.DataFrame(manifest_rows), rename_keys=False, digits=4)
    df.to_csv(out_dir / 'sa_ddpg_table.csv', index=False, float_format='%.4f')
    manifest_df.to_csv(out_dir / 'sa_ddpg_manifest.csv', index=False, float_format='%.4f')
    json_dump(df.to_dict(orient='records'), out_dir / 'sa_ddpg_table.json', normalize_numbers=True, rename_keys=False)
    json_dump(manifest_df.to_dict(orient='records'), out_dir / 'sa_ddpg_manifest.json', normalize_numbers=True, rename_keys=False)
    print(df[['attack', 'clean_reward', 'attack_reward', 'recovery_ratio', 'clean_drop']].to_string(index=False))
    print(f'saved: {out_dir}')




def command_train_agent(args):
    command_baseline_train(args)


def command_collect_clean(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    actor = resolve_actor(args, device)
    policy_tag = actor_source_tag(args.policy_mode, args.actor_path)
    bundle = collect_clean_trajectories(
        arrivals,
        actor,
        args.signals_path,
        device,
        reward_profile=PROFILE_MAP[args.reward_profile],
        episodes=args.episodes,
        max_samples=args.max_samples,
    )
    bundle.metadata.update({'policy_tag': policy_tag})
    save_path = args.save_path or default_clean_path(policy_tag, args.reward_profile, args.clean_dir)
    save_clean_trajectory_dataset(bundle, save_path)
    print('collect-clean complete')
    print(f'dnormal: {save_path}')
    print(f'samples: {bundle.metadata.get("samples", len(bundle.clean_inputs))}')


def command_evaluate_rollout(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    actor = resolve_actor(args, device)
    policy_tag = actor_source_tag(args.policy_mode, args.actor_path)
    expected_base_metadata = _expected_artifact_semantics(
        policy_tag=policy_tag,
        algorithm=args.algorithm,
        scenario=args.scenario,
        epsilon=effective_attack_epsilon(args),
        attack_ratio=args.attack_ratio,
        attack_scope=args.attack_scope,
    )
    attacker = build_attacker_from_args(args, actor, device, arrivals=arrivals)
    resolved_dae_path = getattr(args, 'dae_path', None) or default_eval_dae_load_path(args, policy_tag)
    resolved_detector_path = getattr(args, 'detector_path', None) or default_eval_detector_load_path(args, policy_tag)
    dae = resolve_dae(
        args,
        device,
        expected_metadata=expected_base_metadata,
        default_path=resolved_dae_path,
    )
    detector = resolve_detector(
        args,
        device,
        default_path=resolved_detector_path,
        expected_metadata=expected_base_metadata,
    )
    bundle = evaluate_rollout_bundle(
        arrivals,
        actor,
        args.signals_path,
        device,
        canonical_attack_scenario(args.algorithm, args.scenario),
        attacker=attacker,
        defender=dae,
        detector_model=None if detector is None else detector['model'],
        detector_threshold=None if detector is None else detector['threshold'],
        reward_profile=PROFILE_MAP[args.reward_profile],
        exploration_noise=args.exploration_noise,
        price_threshold=args.price_threshold,
        soc_new_threshold=args.soc_new_threshold,
        soc_rollout_threshold=args.soc_rollout_threshold,
        even_station_target=args.even_station_target,
        odd_station_target=args.odd_station_target,
        attack_ratio=args.attack_ratio,
        attack_scope=args.attack_scope,
        detector_feature_mode='sequence' if detector is None else detector['feature_mode'],
    )
    rows = iter_evaluation_summaries(bundle)
    df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows])
    print(df.to_string(index=False))
    save_root = args.save_dir or DEFAULT_RESULTS_DIR
    out_dir = result_dir_for_rollout(
        save_root,
        policy_tag=actor_source_tag(args.policy_mode, args.actor_path),
        attack_tag=attack_name_fragment(args.algorithm, args.scenario),
        dae_tag=rollout_usage_tag(
            resolved_dae_path,
            resolved_detector_path,
            args.detector_threshold,
        ),
        epsilon=effective_attack_epsilon(args),
        reward_profile=args.reward_profile,
        attack_ratio=args.attack_ratio,
        attack_scope=args.attack_scope,
    )
    save_evaluation_bundle(bundle, out_dir)
    json_dump(
        {
            'clean_no_dae': bundle.clean_summary,
            'attack_no_dae': bundle.attack_summary,
            'clean_dae': bundle.clean_dae_summary,
            'attack_dae': bundle.attack_dae_summary,
            'clean_dae_detector': bundle.clean_dae_detector_summary,
            'attack_dae_detector': bundle.attack_dae_detector_summary,
            'clean_dae_oracle': bundle.clean_dae_oracle_summary,
            'attack_dae_oracle': bundle.attack_dae_oracle_summary,
        },
        out_dir / 'evaluation_metrics.json',
        normalize_numbers=True,
    )
    print(f'saved: {out_dir}')


def command_evaluate_actions(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    actor = resolve_actor(args, device)
    policy_tag = actor_source_tag(args.policy_mode, args.actor_path)
    expected_base_metadata = _expected_artifact_semantics(
        policy_tag=policy_tag,
        algorithm=args.algorithm,
        scenario=args.scenario,
        epsilon=effective_attack_epsilon(args),
        attack_ratio=args.attack_ratio,
        attack_scope=args.attack_scope,
    )
    resolved_dae_path = getattr(args, 'dae_path', None) or default_eval_dae_load_path(args, policy_tag)
    dae = resolve_dae(
        args,
        device,
        expected_metadata=expected_base_metadata,
        default_path=resolved_dae_path,
    )
    pair_path = args.pair_path or default_pair_path(
        actor_source_tag(args.policy_mode, args.actor_path),
        args.algorithm,
        args.scenario,
        effective_attack_epsilon(args),
        args.pair_dir,
        attack_ratio=args.attack_ratio,
        attack_scope=args.attack_scope,
    )
    bundle = load_pair_dataset(pair_path)
    metrics = evaluate_action_dataset(
        bundle.clean_inputs,
        bundle.adv_inputs,
        actor,
        device,
        defender=dae,
        episode_indices=bundle.episode_indices,
        vehicle_ids=bundle.vehicle_ids,
    )
    df = pd.DataFrame([metrics])
    print(df.to_string(index=False))
    save_root = args.save_dir or DEFAULT_RESULTS_DIR
    out_dir = result_dir_for_actions(
        save_root,
        policy_tag=actor_source_tag(args.policy_mode, args.actor_path),
        attack_tag=attack_name_fragment(args.algorithm, args.scenario),
        dae_tag=dae_usage_tag(resolved_dae_path),
        epsilon=effective_attack_epsilon(args),
        source_tag=Path(pair_path).stem,
        attack_ratio=args.attack_ratio,
        attack_scope=args.attack_scope,
    )
    df.to_csv(out_dir / 'action_metrics.csv', index=False, float_format='%.6f')
    json_dump(metrics, out_dir / 'action_metrics.json', normalize_numbers=True, rename_keys=False)
    print(f'saved: {out_dir}')


def command_train_offline_dae_det_temporal_shield(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    actor = resolve_temporal_shield_actor(args, device)
    state_scope = canonical_state_scope(getattr(args, 'state_scope', 'local'))
    if state_scope not in {'local', 'all'}:
        raise ValueError(f'Temporal shield only supports local/all scopes, got: {state_scope}')

    dae_path = Path(getattr(args, 'dae_path', None) or default_temporal_shield_dae_path(state_scope))
    dae = load_dae(dae_path, device)
    detector_path = Path(getattr(args, 'detector_path', None) or default_temporal_shield_detector_path(state_scope))
    detector_artifact = load_detector(detector_path, device)
    if not isinstance(detector_artifact.model, PosteriorBenefitMLPDetector):
        raise ValueError(f'Temporal shield requires a posterior MLP detector artifact, got: {type(detector_artifact.model)!r}')
    detector_threshold = float(args.detector_threshold) if args.detector_threshold is not None else float(detector_artifact.threshold)
    tau_soc_scales = tuple(split_csv_floats(args.tau_soc_scales))
    tau_time_scales = tuple(split_csv_floats(args.tau_time_scales))
    tau_cost_scales = tuple(split_csv_floats(args.tau_cost_scales))

    shield_config, calibration_stats = calibrate_local_temporal_shield(
        arrivals,
        args.signals_path,
        actor,
        device,
        reward_profile=PROFILE_MAP[args.reward_profile],
        calibration_quantile=float(args.calibration_quantile),
        min_tau_soc=float(args.min_tau_soc),
        min_tau_time=float(args.min_tau_time),
        min_tau_cost=float(args.min_tau_cost),
        max_tau_soc=float(args.max_tau_soc),
        max_tau_time=float(args.max_tau_time),
        max_tau_cost=float(args.max_tau_cost),
        state_scope=state_scope,
    )
    tuning_summary = None
    if bool(args.tune_with_attacks):
        raw_attack_bundle_path = getattr(args, 'attack_bundle_path', None)
        if raw_attack_bundle_path is not None:
            attack_bundle_path = Path(raw_attack_bundle_path).expanduser().resolve()
        elif getattr(args, 'actor_path', None) is not None and _torch_artifact_has_critic(getattr(args, 'actor_path', None)):
            attack_bundle_path = Path(getattr(args, 'actor_path')).expanduser().resolve()
        else:
            attack_bundle_path = Path(resolve_default_baseline_bundle_path(args.reward_profile)).expanduser().resolve()
        critic = load_attack_critic('q_function', device, bundle_path=attack_bundle_path)
        shield_config, tuning_summary = tune_temporal_shield_with_attacks(
            arrivals,
            actor,
            args.signals_path,
            device,
            defender=dae,
            detector_model=detector_artifact.model,
            detector_threshold=detector_threshold,
            base_config=shield_config,
            reward_profile=PROFILE_MAP[args.reward_profile],
            clean_drop_limit=float(args.clean_drop_limit),
            tau_soc_scales=tau_soc_scales,
            tau_time_scales=tau_time_scales,
            tau_cost_scales=tau_cost_scales,
            critic=critic,
            seed=int(args.seed),
        )
    metadata = {
        'algorithm': 'temporal_shield',
        'variant': 'gru_vae_dae_posterior_det',
        'baseline_semantics': 'temporal_shield_upgrade',
        'state_scope': state_scope,
        'shield_state_scope': state_scope,
        'reward_profile': str(args.reward_profile),
        'actor_source': actor_source_tag(args.policy_mode, args.actor_path),
        'data_path': str(args.data_path),
        'signals_path': str(args.signals_path),
        'dae_path': str(dae_path),
        'detector_path': str(detector_path),
        'detector_threshold_override': float(detector_threshold),
        'clean_drop_limit': float(args.clean_drop_limit),
    }
    if tuning_summary is not None:
        metadata['attack_tuning'] = tuning_summary
    out_path = Path(getattr(args, 'output_path', None) or default_temporal_shield_bundle_path(state_scope, getattr(args, 'output_dir', None)))
    save_temporal_shield_bundle(
        shield_config,
        out_path,
        metadata=metadata,
        calibration_stats=calibration_stats,
    )
    if tuning_summary is not None:
        selected_row = dict(tuning_summary.get('selected_row') or {})
        print(
            'tuned shield summary: '
            f"tau=({float(shield_config.tau_soc):.6f},{float(shield_config.tau_time):.6f},{float(shield_config.tau_cost):.6f}) "
            f"worst_case_recovery={float(selected_row.get('worst_case_recovery_shield', float('nan'))):.6f} "
            f"mean_recovery={float(selected_row.get('mean_recovery_shield', float('nan'))):.6f} "
            f"clean_drop={float(selected_row.get('clean_drop_shield', float('nan'))):.6f} "
            f"improved={bool(tuning_summary.get('improvement_flag', False))} "
            f"status={str(tuning_summary.get('selection_status', 'unknown'))}"
        )
    print(f'saved temporal shield: {out_path}')
    print(pd.DataFrame([shield_config.to_dict()]).to_string(index=False))


def command_eval_offline_dae_det_temporal_shield(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    actor = resolve_temporal_shield_actor(args, device)
    state_scope = canonical_state_scope(getattr(args, 'state_scope', 'local'))
    if state_scope not in {'local', 'all'}:
        raise ValueError(f'Temporal shield only supports local/all scopes, got: {state_scope}')

    shield_path = Path(getattr(args, 'shield_path', None) or default_temporal_shield_bundle_path(state_scope, getattr(args, 'shield_dir', None)))
    shield_artifact = load_temporal_shield_bundle(shield_path)
    shield_config = shield_artifact.config
    metadata = dict(shield_artifact.metadata or {})

    dae_path = Path(getattr(args, 'dae_path', None) or metadata.get('dae_path') or default_temporal_shield_dae_path(state_scope))
    dae = load_dae(dae_path, device)
    detector_path = Path(getattr(args, 'detector_path', None) or metadata.get('detector_path') or default_temporal_shield_detector_path(state_scope))
    detector_artifact = load_detector(detector_path, device)
    if not isinstance(detector_artifact.model, PosteriorBenefitMLPDetector):
        raise ValueError(f'Temporal shield requires a posterior MLP detector artifact, got: {type(detector_artifact.model)!r}')
    detector_threshold = (
        float(args.detector_threshold)
        if args.detector_threshold is not None
        else float(metadata.get('detector_threshold_override', detector_artifact.threshold))
    )
    critic = load_attack_critic('q_function', device, bundle_path=resolve_attack_bundle_path(args))
    eval_algorithms = tuple(split_csv_strings(args.eval_algorithms))
    rows = eval_temporal_shield_suite(
        arrivals,
        args.signals_path,
        actor,
        dae,
        detector_artifact.model,
        detector_threshold,
        shield_config,
        device,
        reward_profile=PROFILE_MAP[args.reward_profile],
        eval_algorithms=eval_algorithms,
        state_scope=state_scope,
        epsilon_q_pgd=float(args.epsilon_q_pgd),
        epsilon_learned=float(args.epsilon_learned),
        attack_scenario=args.scenario,
        attack_scope=args.attack_scope,
        attack_ratio=float(args.attack_ratio),
        alpha=args.alpha,
        iters=args.iters,
        critic=critic,
        learned_adv_iters=int(args.learned_adv_iters),
        learned_adv_phase_steps=int(args.learned_adv_phase_steps),
        learned_adv_ppo_epochs=int(args.learned_adv_ppo_epochs),
        learned_adv_num_minibatches=int(args.learned_adv_num_minibatches),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_eps=float(args.clip_eps),
        adv_actor_lr=float(args.adv_actor_lr),
        adv_critic_lr=float(args.adv_critic_lr),
        adv_entropy_coeff=float(args.adv_entropy_coeff),
        adversary_hidden_dim=int(args.adversary_hidden_dim),
        seed=int(args.seed),
        print_every=int(args.print_every),
    )
    df = normalize_result_frame(rows)
    if isinstance(df, pd.DataFrame):
        df['dae_path'] = str(dae_path)
        df['detector_path'] = str(detector_path)
        df['detector_threshold'] = float(detector_threshold)
        df['shield_path'] = str(shield_path)
        attack_bundle_path = resolve_attack_bundle_path(args)
        df['attack_bundle_path'] = '' if attack_bundle_path is None else str(attack_bundle_path)
    if isinstance(df, pd.DataFrame) and not df.empty:
        print(df.to_string(index=False))
    output_path = Path(
        getattr(args, 'output_path', None)
        or (Path(getattr(args, 'output_dir', None) or (DEFAULT_RESULTS_DIR / 'offline_dae_det_temporal_shield')) / f'{shield_path.stem}_eval.csv')
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f'saved: {output_path}')


def command_run_unified(args):
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    actor = resolve_actor(args, device)
    policy_tag = actor_source_tag(args.policy_mode, args.actor_path)
    detector_feature_mode = canonical_detector_feature_mode(getattr(args, 'detector_feature_mode', 'posterior'))
    posterior_label_mode = canonical_posterior_label_mode(getattr(args, 'posterior_label_mode', 'benefit'))
    train_algorithms = split_csv_strings(args.train_algorithms)
    eval_algorithms = split_csv_strings(args.eval_algorithms)
    epsilons = split_csv_floats(args.epsilons)
    state_scope = canonical_state_scope(getattr(args, 'state_scope', 'local'))
    scope_default_config = apply_unified_scope_defaults(args, state_scope)
    base_profile_tag = unified_profile_tag(train_algorithms, epsilons)
    profile_tag = args.profile_tag or (base_profile_tag if state_scope == 'local' else f'{base_profile_tag}_scope{state_scope}')
    train_jobs = list(iter_attack_jobs(train_algorithms, ['O'], epsilons))
    eval_jobs = list(iter_attack_jobs(eval_algorithms, split_csv_strings(args.eval_scenarios), epsilons))
    train_attack_tags = [str(job['attack_tag']) for job in train_jobs]

    clean_bundle, clean_path, clean_status = load_or_collect_clean_bundle(
        args,
        arrivals,
        actor,
        device,
        reward_profile=args.collect_reward_profile,
        policy_tag=policy_tag,
        episodes=args.collect_episodes,
        max_samples=args.collect_max_samples,
    )

    pair_path = args.pair_path or default_unified_pair_path(policy_tag, profile_tag, args.pair_dir)
    if args.pair_source in {'load', 'load_or_collect'} and Path(pair_path).exists():
        unified_pair = load_pair_dataset(pair_path)
        _validate_state_scope_alignment('Unified pair dataset', pair_path, unified_pair.metadata, state_scope)
        pair_status = 'loaded'
    else:
        if args.pair_source == 'load':
            raise FileNotFoundError(f'Missing unified pair dataset: {pair_path}')
        source_bundles: list[PairDatasetBundle] = []
        for job in train_jobs:
            critic = load_attack_critic(job['algorithm'], device, bundle_path=resolve_attack_bundle_path(args))
            bundle, _ = collect_pair_bundle_for_job(
                arrivals,
                actor,
                device,
                args.signals_path,
                algorithm=job['algorithm'],
                scenario=job['scenario'],
                epsilon=job['epsilon'],
                seed=args.seed,
                alpha=args.alpha,
                iters=args.iters,
                reward_profile=args.collect_reward_profile,
                episodes=args.collect_episodes,
                max_samples=args.collect_max_samples,
                policy_input_mode='clean',
                price_threshold=args.price_threshold,
                soc_new_threshold=args.soc_new_threshold,
                soc_rollout_threshold=args.soc_rollout_threshold,
                even_station_target=args.even_station_target,
                odd_station_target=args.odd_station_target,
                attack_ratio=1.0,
                attack_scope='obs',
                state_scope=state_scope,
                clean_bundle=clean_bundle,
                critic=critic,
            )
            bundle.metadata.update({'clean_path': str(clean_path), 'clean_status': clean_status, 'attack_tag': job['attack_tag']})
            source_bundles.append(bundle)
        unified_pair = merge_pair_bundles_for_unified(source_bundles, attack_tags=train_attack_tags)
        unified_pair.metadata.update(
            {
                'profile_tag': profile_tag,
                'policy_tag': policy_tag,
                'clean_path': str(clean_path),
                'clean_status': clean_status,
                'epsilons': [float(v) for v in epsilons],
                'state_scope': state_scope,
                'attack_state_scope': state_scope,
                'defense_state_scope': state_scope,
                'attack_state_indices': list(defended_indices_for_scope(state_scope)),
                'scope_default_config': dict(scope_default_config),
            }
        )
        pair_status = 'collected'
        if args.save_pairs:
            save_pair_dataset(unified_pair, pair_path)

    detector_dataset_path = args.detector_dataset_path or default_unified_detector_dataset_path(
        policy_tag,
        profile_tag,
        args.detector_data_dir,
        detector_feature_mode=detector_feature_mode,
        posterior_label_mode=posterior_label_mode,
    )
    dae_path = args.dae_path or default_unified_dae_path(policy_tag, profile_tag, args.dae_dir)
    dae_result = None
    dae_model = None
    dae_status = 'skipped' if args.data_only else None
    if not args.data_only:
        if args.dae_source in {'load', 'load_or_train'} and Path(dae_path).exists():
            dae_model = load_dae(dae_path, device)
            if not _scope_matches_model(dae_model, state_scope):
                raise ValueError(f'DAE state_scope mismatch for {dae_path}: expected {state_scope!r} defended dims {list(defended_indices_for_scope(state_scope))}, got {list(getattr(dae_model, "local_indices", []))}.')
            dae_status = 'loaded'
        else:
            if args.dae_source == 'load':
                raise FileNotFoundError(f'Missing unified DAE model: {dae_path}')
            dae_checkpoint_metric = str(getattr(args, 'dae_checkpoint_metric', 'rollout')).strip().lower()
            dae_validator = None
            dae_select_by = 'loss'
            if dae_checkpoint_metric == 'rollout':
                dae_validator = _make_dae_rollout_checkpoint_validator(
                    args,
                    arrivals,
                    actor,
                    device,
                    train_jobs=train_jobs,
                    reward_profile=args.rollout_reward_profile,
                    clean_penalty=args.dae_checkpoint_clean_penalty,
                )
                dae_select_by = 'dae_checkpoint_score'
            dae_model, dae_result = train_dae_for_job(
                unified_pair,
                actor,
                device,
                epochs=args.dae_epochs,
                batch_size=args.dae_batch_size,
                lr=args.dae_lr,
                lambda_state=args.lambda_state,
                lambda_identity=args.lambda_identity,
                validator=dae_validator,
                val_every=args.dae_val_every,
                select_by=dae_select_by,
                seq_len=args.seq_len,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                num_layers=args.num_layers,
                decoder_hidden_dim=args.decoder_hidden_dim,
                beta_kl=args.beta_kl,
                lambda_robust=args.lambda_robust,
                include_clean_sequences=args.include_clean_sequences,
                state_scope=state_scope,
            )
            dae_status = 'trained'
            if args.save_daes:
                save_dae(
                    dae_model,
                    dae_path,
                    metadata={
                        'artifact_role': 'unified_dae',
                        'profile_tag': profile_tag,
                        'policy_tag': policy_tag,
                        'train_attacks': train_attack_tags,
                        'epsilons': [float(v) for v in epsilons],
                        'pair_path': str(pair_path),
                        'pair_status': pair_status,
                        'attack_trigger_mode': f'candidate_all_{state_scope}_obs',
                        'state_scope': state_scope,
                        'attack_state_scope': state_scope,
                        'defense_state_scope': state_scope,
                        'attack_state_indices': list(defended_indices_for_scope(state_scope)),
                        'scope_default_config': dict(scope_default_config),
                        'epochs': int(args.dae_epochs),
                        'batch_size': int(args.dae_batch_size),
                        'lr': float(args.dae_lr),
                        'lambda_state': float(args.lambda_state),
                        'lambda_identity': float(args.lambda_identity),
                        'dae_checkpoint_metric': str(dae_checkpoint_metric),
                        'dae_val_every': int(args.dae_val_every),
                        'dae_checkpoint_clean_penalty': float(args.dae_checkpoint_clean_penalty),
                        'seq_len': int(args.seq_len),
                        'hidden_dim': int(args.hidden_dim),
                        'latent_dim': int(args.latent_dim),
                        'num_layers': int(args.num_layers),
                        'decoder_hidden_dim': int(args.decoder_hidden_dim),
                        'beta_kl': float(args.beta_kl),
                        'lambda_robust': float(args.lambda_robust),
                        'include_clean_sequences': bool(args.include_clean_sequences),
                        'best_epoch': int(dae_result.best_epoch),
                        'best_metric_name': str(dae_result.best_metric_name),
                        'best_metric_value': float(dae_result.best_metric_value),
                    },
                )
                save_dae_history(dae_result, Path(dae_path).with_name(f'{Path(dae_path).stem}_history.csv'))
    elif detector_feature_mode == 'posterior' and not (args.detector_data_source in {'load', 'load_or_collect'} and Path(detector_dataset_path).exists()):
        if Path(dae_path).exists():
            dae_model = load_dae(dae_path, device)
            if not _scope_matches_model(dae_model, state_scope):
                raise ValueError(f'DAE state_scope mismatch for {dae_path}: expected {state_scope!r} defended dims {list(defended_indices_for_scope(state_scope))}, got {list(getattr(dae_model, "local_indices", []))}.')
            dae_status = 'loaded_for_detector_data'
        else:
            raise FileNotFoundError(f'Posterior detector dataset collection requires an existing unified DAE artifact: {dae_path}')

    if args.detector_data_source in {'load', 'load_or_collect'} and Path(detector_dataset_path).exists():
        detector_dataset = load_detector_dataset(detector_dataset_path)
        validate_detector_dataset_mode(detector_dataset, detector_feature_mode, detector_dataset_path, posterior_label_mode=posterior_label_mode)
        _validate_state_scope_alignment('Detector dataset', detector_dataset_path, detector_dataset.metadata, state_scope)
        detector_dataset_status = 'loaded'
    else:
        if args.detector_data_source == 'load':
            raise FileNotFoundError(f'Missing unified detector dataset: {detector_dataset_path}')
        if detector_feature_mode == 'posterior':
            detector_dataset = posterior_detector_dataset_from_unified_pair(
                unified_pair,
                actor,
                dae_model,
                device,
                profile_tag=profile_tag,
                train_attack_tags=train_attack_tags,
                benefit_margin=args.posterior_benefit_margin,
                benefit_action_weight=args.posterior_benefit_action_weight,
                benefit_state_weight=args.posterior_benefit_state_weight,
                posterior_label_mode=posterior_label_mode,
                use_benefit_sample_weights=args.posterior_use_benefit_weights,
                state_scope=state_scope,
            )
        else:
            detector_dataset = detector_dataset_from_unified_pair(unified_pair, profile_tag=profile_tag, train_attack_tags=train_attack_tags, state_scope=state_scope)
        detector_dataset_status = 'collected'
        if args.save_detector_data:
            save_detector_dataset(detector_dataset, detector_dataset_path)

    if args.data_only:
        output_dir = ensure_dir(args.output_dir / profile_tag if Path(args.output_dir).name != profile_tag else args.output_dir)
        json_dump(
            {
                'profile_tag': profile_tag,
                'policy_tag': policy_tag,
                'train_attacks': train_attack_tags,
                'clean_path': str(clean_path),
                'clean_status': clean_status,
                'min_clean_samples': int(_resolve_clean_requirements(args, max_samples=args.collect_max_samples)[0]),
                'min_clean_groups': int(_resolve_clean_requirements(args, max_samples=args.collect_max_samples)[1]),
                'unified_pair_path': str(pair_path),
                'unified_pair_status': pair_status,
                'unified_detector_dataset_path': str(detector_dataset_path),
                'unified_detector_dataset_status': detector_dataset_status,
                'detector_feature_mode': detector_feature_mode,
                'posterior_label_mode': posterior_label_mode if detector_feature_mode == 'posterior' else None,
                'data_only': True,
                'attack_trigger_mode': f'candidate_all_{state_scope}_obs',
                'state_scope': state_scope,
                'attack_state_scope': state_scope,
                'defense_state_scope': state_scope,
                'attack_state_indices': list(defended_indices_for_scope(state_scope)),
                'scope_default_config': dict(scope_default_config),
            },
            output_dir / 'unified_dataset_artifacts.json',
            normalize_numbers=True,
            rename_keys=False,
        )
        print(
            pd.DataFrame(
                [
                    {
                        'profile_tag': profile_tag,
                        'clean_status': clean_status,
                        'pair_status': pair_status,
                        'detector_dataset_status': detector_dataset_status,
                    }
                ]
            ).to_string(index=False)
        )
        print(f'dnormal: {clean_path}')
        print(f'unified_pair: {pair_path}')
        print(f'detector_dataset: {detector_dataset_path}')
        print(f'saved: {output_dir}')
        return

    detector_path = args.detector_path or default_unified_detector_path(
        policy_tag,
        profile_tag,
        args.detector_dir,
        detector_feature_mode=detector_feature_mode,
        posterior_label_mode=posterior_label_mode,
    )
    detector_train_result = None
    detector_selection_result = None
    if args.detector_source in {'load', 'load_or_tune'} and Path(detector_path).exists():
        detector_artifact = load_detector(detector_path, device)
        detector_model = detector_artifact.model
        detector_threshold = float(detector_artifact.threshold)
        _validate_state_scope_alignment('Detector artifact', detector_path, detector_artifact.metadata, state_scope)
        artifact_feature_mode = canonical_detector_feature_mode((detector_artifact.metadata or {}).get('detector_feature_mode', detector_feature_mode))
        if artifact_feature_mode != detector_feature_mode:
            raise ValueError(f'Detector artifact mode mismatch for {detector_path}: expected {detector_feature_mode!r}, got {artifact_feature_mode!r}.')
        if detector_feature_mode == 'posterior':
            artifact_label_mode = canonical_posterior_label_mode((detector_artifact.metadata or {}).get('posterior_label_mode', 'benefit'))
            if artifact_label_mode != posterior_label_mode:
                raise ValueError(f'Posterior detector artifact label mode mismatch for {detector_path}: expected {posterior_label_mode!r}, got {artifact_label_mode!r}.')
        detector_status = 'loaded'
    else:
        if args.detector_source == 'load':
            raise FileNotFoundError(f'Missing unified detector model: {detector_path}')
        detector_model, detector_train_result = train_detector_from_bundle(
            detector_dataset,
            actor,
            dae_model,
            device,
            epochs=args.detector_epochs,
            batch_size=args.detector_batch_size,
            lr=args.detector_lr,
            hidden_dim=args.detector_hidden_dim,
            dropout=0.1 if detector_feature_mode == 'posterior' else 0.0,
            val_ratio=args.detector_val_ratio,
            detector_temporal=True,
            detector_feature_mode=detector_feature_mode,
            seed=args.seed,
            latent_dim=args.detector_latent_dim,
            num_layers=args.detector_num_layers,
            beta_kl=args.detector_beta_kl,
            seq_len=args.detector_seq_len,
            state_scope=state_scope,
        )
        threshold_attacker = None
        threshold_scenario = 'O'
        if detector_feature_mode == 'posterior' and train_jobs:
            threshold_job = train_jobs[0]
            obs_low, obs_high = resolve_attack_obs_bounds(arrivals, args.signals_path)
            threshold_critic = load_attack_critic(threshold_job['algorithm'], device, bundle_path=resolve_attack_bundle_path(args))
            threshold_attacker = build_attacker(
                actor,
                device,
                algorithm=threshold_job['algorithm'],
                epsilon=float(threshold_job['epsilon']),
                alpha=args.alpha,
                iters=args.iters,
                seed=args.seed,
                obs_low=obs_low,
                obs_high=obs_high,
                critic=threshold_critic,
                attack_state_scope=state_scope,
            )
            threshold_scenario = canonical_attack_scenario(threshold_job['algorithm'], threshold_job['scenario'])
        detector_selection_result = select_detector_threshold(
            detector_dataset,
            detector_model,
            arrivals,
            actor,
            args.signals_path,
            device,
            threshold_scenario,
            attacker=threshold_attacker,
            defender=dae_model,
            reward_profile=PROFILE_MAP[args.rollout_reward_profile],
            grid_size=args.detector_threshold_grid_size,
            clean_reward_floor_ratio=args.detector_clean_reward_floor_ratio,
            detector_feature_mode=detector_feature_mode,
            exploration_noise=args.exploration_noise,
            price_threshold=args.price_threshold,
            soc_new_threshold=args.soc_new_threshold,
            soc_rollout_threshold=args.soc_rollout_threshold,
            even_station_target=args.even_station_target,
            odd_station_target=args.odd_station_target,
            attack_ratio=1.0,
            attack_scope='obs',
        )
        detector_threshold = float(detector_selection_result.threshold)
        detector_status = 'tuned'
        if args.save_detectors:
            save_detector(
                detector_model,
                detector_path,
                threshold=detector_threshold,
                metadata={
                    'artifact_role': 'unified_detector',
                    'detector_feature_mode': detector_feature_mode,
                    'detector_mode': detector_dataset_mode_for_feature_mode(detector_feature_mode),
                    'posterior_label_mode': posterior_label_mode if detector_feature_mode == 'posterior' else None,
                    'profile_tag': profile_tag,
                    'policy_tag': policy_tag,
                    'train_attacks': train_attack_tags,
                    'epsilons': [float(v) for v in epsilons],
                    'detector_dataset_path': str(detector_dataset_path),
                    'detector_dataset_status': detector_dataset_status,
                    'dae_path': str(dae_path),
                    'attack_trigger_mode': f'candidate_all_{state_scope}_obs',
                    'state_scope': state_scope,
                    'attack_state_scope': state_scope,
                    'defense_state_scope': state_scope,
                    'attack_state_indices': list(defended_indices_for_scope(state_scope)),
                    'scope_default_config': dict(scope_default_config),
                },
            )
            save_detector_history(detector_train_result, Path(detector_path).with_name(f'{Path(detector_path).stem}_train_history.csv'))
            pd.DataFrame(detector_selection_result.history_rows).pipe(normalize_result_frame, rename_keys=False).to_csv(Path(detector_path).with_name(f'{Path(detector_path).stem}_threshold_history.csv'), index=False, float_format='%.6f')

    posterior_route_summary = None
    if detector_feature_mode == 'posterior':
        posterior_route_summary = summarize_posterior_route_decisions(
            detector_dataset,
            detector_model,
            detector_threshold,
            actor,
            device,
        )

    rows = []
    manifest_rows = []
    for job in eval_jobs:
        critic = load_attack_critic(job['algorithm'], device, bundle_path=resolve_attack_bundle_path(args))
        eval_bundle, attacker = collect_pair_bundle_for_job(
            arrivals,
            actor,
            device,
            args.signals_path,
            algorithm=job['algorithm'],
            scenario=job['scenario'],
            epsilon=job['epsilon'],
            seed=args.seed,
            alpha=args.alpha,
            iters=args.iters,
            reward_profile=args.collect_reward_profile,
            episodes=args.collect_episodes,
            max_samples=args.collect_max_samples,
            policy_input_mode='clean',
            price_threshold=args.price_threshold,
            soc_new_threshold=args.soc_new_threshold,
            soc_rollout_threshold=args.soc_rollout_threshold,
            even_station_target=args.even_station_target,
            odd_station_target=args.odd_station_target,
            attack_ratio=1.0,
            attack_scope='obs',
            state_scope=state_scope,
            clean_bundle=clean_bundle,
            critic=critic,
        )
        action_attack = evaluate_action_dataset(eval_bundle.clean_inputs, eval_bundle.adv_inputs, actor, device, defender=None, episode_indices=eval_bundle.episode_indices, vehicle_ids=eval_bundle.vehicle_ids)
        action_defend = evaluate_action_dataset(eval_bundle.clean_inputs, eval_bundle.adv_inputs, actor, device, defender=dae_model, episode_indices=eval_bundle.episode_indices, vehicle_ids=eval_bundle.vehicle_ids)
        rollout = evaluate_rollout_bundle(
            arrivals,
            actor,
            args.signals_path,
            device,
            canonical_attack_scenario(job['algorithm'], job['scenario']),
            attacker=attacker,
            defender=dae_model,
            detector_model=detector_model,
            detector_threshold=detector_threshold,
            reward_profile=PROFILE_MAP[args.rollout_reward_profile],
            exploration_noise=args.exploration_noise,
            price_threshold=args.price_threshold,
            soc_new_threshold=args.soc_new_threshold,
            soc_rollout_threshold=args.soc_rollout_threshold,
            even_station_target=args.even_station_target,
            odd_station_target=args.odd_station_target,
            attack_ratio=1.0,
            attack_scope='obs',
            detector_feature_mode=detector_feature_mode,
        )
        attack_tag = str(job['attack_tag'])
        row = {
            'attack': attack_tag,
            'algorithm': job['algorithm'],
            'scenario': job['scenario'],
            'epsilon': float(job['epsilon']),
            'profile_tag': profile_tag,
            'state_scope': state_scope,
            'detector_feature_mode': detector_feature_mode,
            'posterior_label_mode': posterior_label_mode if detector_feature_mode == 'posterior' else None,
            'unified_dae_path': str(dae_path),
            'unified_detector_path': str(detector_path),
            'detector_threshold': float(detector_threshold),
            **{f'action_{k}': v for k, v in action_attack.items()},
            **{f'defended_{k}': v for k, v in action_defend.items() if k != 'sample_count'},
            'rollout_clean_reward': float(rollout.clean_summary['ep_reward']),
            'rollout_attack_reward': float(rollout.attack_summary['ep_reward']),
            'rollout_clean_dae_reward': None if rollout.clean_dae_summary is None else float(rollout.clean_dae_summary['ep_reward']),
            'rollout_attack_dae_reward': None if rollout.attack_dae_summary is None else float(rollout.attack_dae_summary['ep_reward']),
            'rollout_clean_dae_detector_reward': None if rollout.clean_dae_detector_summary is None else float(rollout.clean_dae_detector_summary['ep_reward']),
            'rollout_attack_dae_detector_reward': None if rollout.attack_dae_detector_summary is None else float(rollout.attack_dae_detector_summary['ep_reward']),
            'rollout_clean_dae_oracle_reward': None if rollout.clean_dae_oracle_summary is None else float(rollout.clean_dae_oracle_summary['ep_reward']),
            'rollout_attack_dae_oracle_reward': None if rollout.attack_dae_oracle_summary is None else float(rollout.attack_dae_oracle_summary['ep_reward']),
            'rollout_clean_dae_route_rate': None if rollout.clean_dae_summary is None else float(rollout.clean_dae_summary.get('route_rate', 0.0)),
            'rollout_attack_dae_route_rate': None if rollout.attack_dae_summary is None else float(rollout.attack_dae_summary.get('route_rate', 0.0)),
            'rollout_clean_dae_detector_route_rate': None if rollout.clean_dae_detector_summary is None else float(rollout.clean_dae_detector_summary.get('route_rate', 0.0)),
            'rollout_attack_dae_detector_route_rate': None if rollout.attack_dae_detector_summary is None else float(rollout.attack_dae_detector_summary.get('route_rate', 0.0)),
            'rollout_clean_dae_oracle_route_rate': None if rollout.clean_dae_oracle_summary is None else float(rollout.clean_dae_oracle_summary.get('route_rate', 0.0)),
            'rollout_attack_dae_oracle_route_rate': None if rollout.attack_dae_oracle_summary is None else float(rollout.attack_dae_oracle_summary.get('route_rate', 0.0)),
            'rollout_attack_obs_rate': float(rollout.attack_summary.get('attack_obs_rate', 0.0)),
            'rollout_clean_exit_vio': int(rollout.clean_summary['exit_vio']),
            'rollout_attack_exit_vio': int(rollout.attack_summary['exit_vio']),
            'rollout_clean_dae_exit_vio': None if rollout.clean_dae_summary is None else int(rollout.clean_dae_summary['exit_vio']),
            'rollout_attack_dae_exit_vio': None if rollout.attack_dae_summary is None else int(rollout.attack_dae_summary['exit_vio']),
            'rollout_clean_dae_detector_exit_vio': None if rollout.clean_dae_detector_summary is None else int(rollout.clean_dae_detector_summary['exit_vio']),
            'rollout_attack_dae_detector_exit_vio': None if rollout.attack_dae_detector_summary is None else int(rollout.attack_dae_detector_summary['exit_vio']),
            'rollout_clean_dae_oracle_exit_vio': None if rollout.clean_dae_oracle_summary is None else int(rollout.clean_dae_oracle_summary['exit_vio']),
            'rollout_attack_dae_oracle_exit_vio': None if rollout.attack_dae_oracle_summary is None else int(rollout.attack_dae_oracle_summary['exit_vio']),
            'rollout_attack_dae_ep_r1': None if rollout.attack_dae_summary is None else float(rollout.attack_dae_summary['ep_r1']),
            'rollout_attack_dae_ep_r2': None if rollout.attack_dae_summary is None else float(rollout.attack_dae_summary['ep_r2']),
            'rollout_attack_dae_ep_r3': None if rollout.attack_dae_summary is None else float(rollout.attack_dae_summary['ep_r3']),
            'rollout_attack_dae_detector_ep_r1': None if rollout.attack_dae_detector_summary is None else float(rollout.attack_dae_detector_summary['ep_r1']),
            'rollout_attack_dae_detector_ep_r2': None if rollout.attack_dae_detector_summary is None else float(rollout.attack_dae_detector_summary['ep_r2']),
            'rollout_attack_dae_detector_ep_r3': None if rollout.attack_dae_detector_summary is None else float(rollout.attack_dae_detector_summary['ep_r3']),
            'rollout_attack_dae_oracle_ep_r1': None if rollout.attack_dae_oracle_summary is None else float(rollout.attack_dae_oracle_summary['ep_r1']),
            'rollout_attack_dae_oracle_ep_r2': None if rollout.attack_dae_oracle_summary is None else float(rollout.attack_dae_oracle_summary['ep_r2']),
            'rollout_attack_dae_oracle_ep_r3': None if rollout.attack_dae_oracle_summary is None else float(rollout.attack_dae_oracle_summary['ep_r3']),
        }
        row['rollout_defended_reward'] = row['rollout_attack_dae_reward']
        row['rollout_defended_exit_vio'] = row['rollout_attack_dae_exit_vio']
        row['rollout_defended_ep_r1'] = row['rollout_attack_dae_ep_r1']
        row['rollout_defended_ep_r2'] = row['rollout_attack_dae_ep_r2']
        row['rollout_defended_ep_r3'] = row['rollout_attack_dae_ep_r3']
        rows.append(row)
        manifest_rows.append(
            {
                'attack': attack_tag,
                'epsilon': float(job['epsilon']),
                'profile_tag': profile_tag,
                'state_scope': state_scope,
                'train_attacks': ','.join(train_attack_tags),
                'pair_path': str(pair_path),
                'pair_status': pair_status,
                'dae_path': str(dae_path),
                'dae_status': dae_status,
                'detector_dataset_path': str(detector_dataset_path),
                'detector_dataset_status': detector_dataset_status,
                'detector_path': str(detector_path),
                'detector_status': detector_status,
                'detector_feature_mode': detector_feature_mode,
                'posterior_label_mode': posterior_label_mode if detector_feature_mode == 'posterior' else None,
            }
        )

    output_dir = ensure_dir(args.output_dir / profile_tag if Path(args.output_dir).name != profile_tag else args.output_dir)
    raw_df, paper_df = save_matrix_outputs(pd.DataFrame(rows), output_dir, manifest_rows=manifest_rows)
    if posterior_route_summary is not None:
        pd.DataFrame([posterior_route_summary]).pipe(normalize_result_frame, rename_keys=False).to_csv(output_dir / 'posterior_route_summary.csv', index=False, float_format='%.6f')
        json_dump(posterior_route_summary, output_dir / 'posterior_route_summary.json', normalize_numbers=True, rename_keys=False)
    json_dump(
        {
            'profile_tag': profile_tag,
            'policy_tag': policy_tag,
            'train_attacks': train_attack_tags,
            'eval_attacks': [str(job['attack_tag']) for job in eval_jobs],
            'clean_path': str(clean_path),
            'clean_status': clean_status,
            'unified_pair_path': str(pair_path),
            'unified_pair_status': pair_status,
            'unified_dae_path': str(dae_path),
            'unified_dae_status': dae_status,
            'unified_detector_dataset_path': str(detector_dataset_path),
            'unified_detector_dataset_status': detector_dataset_status,
            'unified_detector_path': str(detector_path),
            'unified_detector_status': detector_status,
            'detector_threshold': float(detector_threshold),
            'detector_feature_mode': detector_feature_mode,
            'posterior_label_mode': posterior_label_mode if detector_feature_mode == 'posterior' else None,
            'posterior_route_summary': posterior_route_summary,
            'attack_trigger_mode': f'candidate_all_{state_scope}_obs',
            'state_scope': state_scope,
            'attack_state_scope': state_scope,
            'defense_state_scope': state_scope,
            'attack_state_indices': list(defended_indices_for_scope(state_scope)),
            'scope_default_config': dict(scope_default_config),
        },
        output_dir / 'unified_artifacts.json',
        normalize_numbers=True,
        rename_keys=False,
    )
    print(raw_df[[col for col in ['attack', 'epsilon', 'rollout_clean_reward', 'rollout_attack_reward', 'rollout_attack_dae_detector_reward'] if col in raw_df.columns]].to_string(index=False))
    print(f'saved: {output_dir}')






def resolve_wocar_actor_save_path(
    output_dir: Path,
    actor_model_name: str | None,
    episodes: int,
    seed: int,
    reward_profile: str,
    state_scope: str,
    target_lambda: float,
    actor_beta: float,
    actor_q_weight: float,
    actor_reg_weight: float,
    actor_reg_weight_mode: str,
    actor_reg_weight_clip: float,
    train_attacks: list[str] | tuple[str, ...] | str | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if actor_model_name:
        return output_dir / actor_model_name
    scope = _filename_token(canonical_state_scope(state_scope))
    reward_token = _filename_token(reward_profile)
    attack_token = _attack_filename_token(train_attacks)
    lambda_token = _float_filename_token(target_lambda)
    beta_token = _float_filename_token(actor_beta)
    q_token = _float_filename_token(actor_q_weight)
    reg_token = _float_filename_token(actor_reg_weight)
    mode_token = _filename_token(actor_reg_weight_mode)
    clip_token = _float_filename_token(actor_reg_weight_clip)
    return output_dir / (
        f'wocar_{reward_token}_lam{lambda_token}_beta{beta_token}_'
        f'qactor{q_token}_reg{reg_token}_{mode_token}_clip{clip_token}_'
        f'{attack_token}_ep{int(episodes)}_seed{int(seed)}_{scope}.pt'
    )



def command_train_wocar(args):
    """Train the migrated WocaR baseline.

    This is the former online_wocar_v4_qgap_clip1 implementation, exposed in this
    project simply as WocaR.
    """
    torch.set_num_threads(args.num_threads)
    set_all_seeds(args.seed)
    device = prepare_device(args.cuda)
    arrivals = get_arrivals(args.data_path, seed=args.seed, max_sessions=args.max_sessions)
    resolved_validation_baseline_bundle_path = args.wocar_validation_baseline_bundle_path
    if int(args.wocar_validation_every) > 0 and resolved_validation_baseline_bundle_path is None:
        resolved_validation_baseline_bundle_path = resolve_default_baseline_bundle_path(args.reward_profile)
    resolved_resume_bundle_path = args.resume_bundle_path
    resolved_init_actor_path = args.init_actor_path
    if resolved_resume_bundle_path is None and resolved_init_actor_path is None and bool(args.wocar_baseline_warmstart):
        candidate = args.wocar_init_bundle_path or resolve_default_baseline_bundle_path(args.reward_profile)
        candidate = Path(candidate).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f'WocaR baseline warm-start bundle not found: {candidate}')
        resolved_resume_bundle_path = candidate
    elif resolved_resume_bundle_path is None and args.wocar_init_bundle_path is not None:
        resolved_resume_bundle_path = Path(args.wocar_init_bundle_path).expanduser().resolve()
    train_attacks = split_csv_strings(args.wocar_train_attacks)
    agent, history = train_wocar_agent(
        arrivals,
        args.signals_path,
        device,
        seed=args.seed,
        episodes=args.episodes,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        update_every=args.wocar_update_every,
        exploration_noise=args.exploration_noise,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        print_every=args.print_every,
        init_actor_path=resolved_init_actor_path,
        resume_bundle_path=resolved_resume_bundle_path,
        reward_profile=PROFILE_MAP[args.reward_profile],
        train_attacks=train_attacks,
        epsilon=args.wocar_epsilon,
        alpha=args.wocar_alpha,
        steps=args.wocar_steps,
        noise_std=args.wocar_noise_std,
        state_scope=args.wocar_state_scope,
        target_lambda=args.wocar_target_lambda,
        actor_beta=args.wocar_actor_beta,
        actor_q_weight=args.wocar_actor_q_weight,
        actor_reg_weight=args.wocar_actor_reg_weight,
        actor_reg_weight_mode=args.wocar_actor_reg_weight_mode,
        actor_reg_weight_clip=args.wocar_actor_reg_weight_clip,
        validation_every=args.wocar_validation_every,
        validation_attacks=split_csv_strings(args.wocar_validation_attacks) if args.wocar_validation_attacks else None,
        validation_baseline_bundle_path=resolved_validation_baseline_bundle_path,
        validation_clean_drop_hard_cap=args.wocar_validation_clean_drop_hard_cap,
        validation_clean_drop_weight=args.wocar_validation_clean_drop_weight,
    )
    model_path = resolve_wocar_actor_save_path(
        args.output_dir,
        args.actor_model_name,
        args.episodes,
        args.seed,
        args.reward_profile,
        args.wocar_state_scope,
        args.wocar_target_lambda,
        args.wocar_actor_beta,
        args.wocar_actor_q_weight,
        args.wocar_actor_reg_weight,
        args.wocar_actor_reg_weight_mode,
        args.wocar_actor_reg_weight_clip,
        train_attacks,
    )
    save_actor(agent.actor, model_path)
    history_path = model_path.with_name(f'{model_path.stem}_history.csv')
    save_train_history(history, history_path)
    validation_history_path = None
    if getattr(history, 'validation_rows', None):
        validation_history_path = model_path.with_name(f'{model_path.stem}_validation.csv')
        normalize_result_frame(pd.DataFrame(history.validation_rows), rename_keys=False).to_csv(
            validation_history_path, index=False, float_format='%.4f'
        )
    bundle_path = args.bundle_path or model_path.with_name(f'{model_path.stem}_bundle.pt')
    metadata = {
        'algorithm': 'wocar',
        'policy_tag': 'wocar',
        'source_variant': 'wocar',
        'source_provenance': 'online_wocar_v4_qgap_clip1',
        'source_algorithm': 'online_wocar_v4',
        'wocar_algorithm_variant': 'v4_clean_replay_q_gap_weighted_state_regularization',
        'episodes': int(args.episodes),
        'seed': int(args.seed),
        'buffer_size': int(args.buffer_size),
        'batch_size': int(args.batch_size),
        'learning_starts': int(args.learning_starts),
        'wocar_update_every': int(args.wocar_update_every),
        'exploration_noise': float(args.exploration_noise),
        'gamma': float(args.gamma),
        'tau': float(args.tau),
        'actor_lr': float(args.actor_lr),
        'critic_lr': float(args.critic_lr),
        'reward_profile': str(args.reward_profile),
        'wocar_train_attacks': train_attacks,
        'train_attacks': train_attacks,
        'state_scope': str(args.wocar_state_scope),
        'train_scope': str(args.wocar_state_scope),
        'wocar_state_scope': str(args.wocar_state_scope),
        'wocar_target_lambda': float(args.wocar_target_lambda),
        'wocar_actor_beta': float(args.wocar_actor_beta),
        'wocar_actor_q_weight': float(args.wocar_actor_q_weight),
        'wocar_actor_reg_weight': float(args.wocar_actor_reg_weight),
        'wocar_actor_reg_weight_mode': str(args.wocar_actor_reg_weight_mode),
        'wocar_actor_reg_weight_clip': float(args.wocar_actor_reg_weight_clip),
        'wocar_epsilon': float(args.wocar_epsilon),
        'wocar_alpha': None if args.wocar_alpha is None else float(args.wocar_alpha),
        'wocar_steps': None if args.wocar_steps is None else int(args.wocar_steps),
        'wocar_noise_std': float(args.wocar_noise_std),
        'wocar_validation_every': int(args.wocar_validation_every),
        'wocar_validation_attacks': split_csv_strings(args.wocar_validation_attacks) if args.wocar_validation_attacks else train_attacks,
        'wocar_validation_baseline_bundle_path': None if resolved_validation_baseline_bundle_path is None else str(resolved_validation_baseline_bundle_path),
        'wocar_validation_clean_drop_hard_cap': float(args.wocar_validation_clean_drop_hard_cap),
        'wocar_validation_clean_drop_weight': float(args.wocar_validation_clean_drop_weight),
        'wocar_best_validation': getattr(agent, 'best_validation', None),
        'rollout_attack_prob': 0.0,
        'replay_is_clean': True,
        'wocar_baseline_warmstart': bool(args.wocar_baseline_warmstart),
        'wocar_init_bundle_path': None if args.wocar_init_bundle_path is None else str(args.wocar_init_bundle_path),
        'init_actor_path': None if resolved_init_actor_path is None else str(resolved_init_actor_path),
        'resume_bundle_path': None if resolved_resume_bundle_path is None else str(resolved_resume_bundle_path),
    }
    save_wocar_bundle(agent, bundle_path, metadata=metadata)
    print('train-wocar complete')
    print(f'actor:   {model_path}')
    print(f'bundle:  {bundle_path}')
    print(f'history: {history_path}')
    if validation_history_path is not None:
        print(f'validation: {validation_history_path}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='EVC unified defense CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    baseline_cmd = subparsers.add_parser('baseline-train', help='Train or fine-tune the baseline actor.')
    add_runtime_args(baseline_cmd)
    baseline_cmd.add_argument('--episodes', type=int, default=20)
    baseline_cmd.add_argument('--buffer-size', type=int, default=100000)
    baseline_cmd.add_argument('--batch-size', type=int, default=256)
    baseline_cmd.add_argument('--learning-starts', type=int, default=2500)
    baseline_cmd.add_argument('--exploration-noise', type=float, default=1.0)
    baseline_cmd.add_argument('--gamma', type=float, default=0.9)
    baseline_cmd.add_argument('--tau', type=float, default=0.005)
    baseline_cmd.add_argument('--actor-lr', type=float, default=3e-4)
    baseline_cmd.add_argument('--critic-lr', type=float, default=3e-4)
    baseline_cmd.add_argument('--print-every', type=int, default=1)
    baseline_cmd.add_argument('--init-actor-path', type=str2path, default=None)
    baseline_cmd.add_argument('--resume-bundle-path', type=str2path, default=None)
    baseline_cmd.add_argument('--freeze-actor', action=argparse.BooleanOptionalAction, default=False)
    baseline_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    baseline_cmd.add_argument('--output-dir', type=str2path, default=PROJECT_ROOT / 'models' / 'baseline')
    baseline_cmd.add_argument('--actor-model-name', type=str, default=None)
    baseline_cmd.add_argument('--bundle-name', type=str, default=None)
    baseline_cmd.add_argument('--bundle-path', type=str2path, default=None)
    baseline_cmd.add_argument('--publish-policy', action=argparse.BooleanOptionalAction, default=False)
    baseline_cmd.set_defaults(func=command_baseline_train)

    train_agent_cmd = subparsers.add_parser('train-agent', help='Compatibility alias for baseline-train.')
    for action in baseline_cmd._actions[1:]:
        if action.dest not in {'help'}:
            train_agent_cmd._add_action(action)
    train_agent_cmd.set_defaults(func=command_train_agent)

    sa_ddpg_cmd = subparsers.add_parser('train-sa-ddpg', help='Train the frozen SA-DDPG robust actor baseline. Final tuned setting is documented in docs/sa_ddpg_final.md.')
    add_runtime_args(sa_ddpg_cmd)
    sa_ddpg_cmd.add_argument('--episodes', type=int, default=20)
    sa_ddpg_cmd.add_argument('--buffer-size', type=int, default=100000)
    sa_ddpg_cmd.add_argument('--batch-size', type=int, default=256)
    sa_ddpg_cmd.add_argument('--learning-starts', type=int, default=2500)
    sa_ddpg_cmd.add_argument('--exploration-noise', type=float, default=1.0)
    sa_ddpg_cmd.add_argument('--gamma', type=float, default=0.9)
    sa_ddpg_cmd.add_argument('--tau', type=float, default=0.005)
    sa_ddpg_cmd.add_argument('--actor-lr', type=float, default=3e-4)
    sa_ddpg_cmd.add_argument('--critic-lr', type=float, default=3e-4)
    sa_ddpg_cmd.add_argument('--print-every', type=int, default=1)
    sa_ddpg_cmd.add_argument('--init-actor-path', type=str2path, default=None)
    sa_ddpg_cmd.add_argument('--resume-bundle-path', type=str2path, default=None)
    sa_ddpg_cmd.add_argument('--sa-init-bundle-path', type=str2path, default=None, help='Optional baseline actor+critic bundle used for fair SA-DDPG warm-start.')
    sa_ddpg_cmd.add_argument('--sa-baseline-warmstart', action=argparse.BooleanOptionalAction, default=True, help='Warm-start SA-DDPG from the baseline actor+critic bundle when no explicit init path is given.')
    sa_ddpg_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    sa_ddpg_cmd.add_argument('--sa-train-attacks', type=str, default='opposite_pgd,q_function', help='Comma-separated online training attacks cycled by episode.')
    sa_ddpg_cmd.add_argument('--sa-epsilon', type=float, default=0.1, help='L-inf perturbation budget for the online state adversary.')
    sa_ddpg_cmd.add_argument('--sa-alpha', type=float, default=None, help='Optional adversary step size override. Defaults to the selected attack family.')
    sa_ddpg_cmd.add_argument('--sa-steps', type=int, default=None, help='Optional adversary step-count override. Defaults to the selected attack family.')
    sa_ddpg_cmd.add_argument('--sa-objective', choices=['q_function', 'min_q', 'action'], default='q_function', help='Fallback adversary objective when a training family does not define its own white-box loss.')
    sa_ddpg_cmd.add_argument('--sa-noise-std', type=float, default=0.0, help='Optional Gaussian noise added after each adversary step for an SGLD-style inner loop.')
    sa_ddpg_cmd.add_argument('--sa-state-scope', choices=['local', 'global', 'all'], default='all', help='State dimensions the online adversary may perturb during SA-DDPG training.')
    sa_ddpg_cmd.add_argument('--sa-actor-reg-weight', type=float, default=0.1, help='Weight for clean-vs-attacked actor consistency regularization during SA-DDPG updates.')
    sa_ddpg_cmd.add_argument('--sa-mixed-update-attacks', action=argparse.BooleanOptionalAction, default=True, help='Mix the configured SA-DDPG attacks inside each replay update batch instead of using only the episode attack family.')
    sa_ddpg_cmd.add_argument('--sa-anchor-actor-path', type=str2path, default=None, help='Baseline actor used as the clean-action distillation anchor. Defaults to the project baseline actor when --sa-anchor-reg-weight > 0.')
    sa_ddpg_cmd.add_argument('--sa-anchor-reg-weight', type=float, default=0.1, help='Weight for baseline clean-action distillation on clean and attacked observations.')
    sa_ddpg_cmd.add_argument('--sa-anchor-clean-weight', type=float, default=1.0, help='Relative weight for the clean-observation part of the baseline anchor loss.')
    sa_ddpg_cmd.add_argument('--sa-clean-policy-weight', type=float, default=0.0, help='Extra clean-observation actor objective weight.')
    sa_ddpg_cmd.add_argument('--sa-risk-weight-scale', type=float, default=0.0, help='Scale for SOC/time risk weighting in clean-preserving anchor and consistency losses. 0 disables risk weighting.')
    sa_ddpg_cmd.add_argument('--sa-risk-weight-max', type=float, default=3.0, help='Maximum per-sample risk weight for clean-preserving losses.')
    sa_ddpg_cmd.add_argument('--sa-risk-target-soc', type=float, default=None, help='SOC target used by risk weighting. Defaults to the reward profile exit target.')
    sa_ddpg_cmd.add_argument('--sa-validation-every', type=int, default=1, help='Run deterministic validation rollout every N episodes and restore the best checkpoint. Use 0 to disable.')
    sa_ddpg_cmd.add_argument('--sa-validation-attacks', type=str, default=None, help='Comma-separated attacks used for validation checkpoint selection. Defaults to --sa-train-attacks.')
    sa_ddpg_cmd.add_argument('--sa-validation-baseline-bundle-path', type=str2path, default=None, help='Baseline actor+critic bundle used to score SA-DDPG recovery during validation. Defaults to the project baseline bundle.')
    sa_ddpg_cmd.add_argument('--sa-validation-clean-drop-weight', type=float, default=0.3, help='Penalty weight for clean-drop ratio in SA-DDPG validation checkpoint score.')
    sa_ddpg_cmd.add_argument('--sa-validation-clean-drop-budget', type=float, default=0.0, help='If positive, checkpoint scoring penalizes clean drop relative to this absolute reward budget instead of relative clean-drop ratio.')
    sa_ddpg_cmd.add_argument('--sa-validation-clean-drop-hard-cap', type=float, default=0.0, help='If positive, validation checkpoints with clean drop above this reward cap are made ineligible for best selection.')
    sa_ddpg_cmd.add_argument('--sa-validation-clean-exit-weight', type=float, default=0.0, help='Checkpoint score penalty weight for clean exit-violation increase over the DDPG baseline.')
    sa_ddpg_cmd.add_argument('--sa-rollout-attack-prob-start', type=float, default=0.3, help='Curriculum start probability for perturbing rollout observations after warmup.')
    sa_ddpg_cmd.add_argument('--sa-rollout-attack-prob', type=float, default=1.0, help='Curriculum final probability for perturbing rollout observations.')
    sa_ddpg_cmd.add_argument('--sa-update-attack-prob-start', type=float, default=0.5, help='Curriculum start probability for perturbing replay observations after warmup.')
    sa_ddpg_cmd.add_argument('--sa-update-attack-prob', type=float, default=1.0, help='Curriculum final probability for perturbing replay observations.')
    sa_ddpg_cmd.add_argument('--sa-curriculum-steps', type=int, default=30000, help='Number of environment steps used to ramp attack probabilities from start to final values after warmup.')
    sa_ddpg_cmd.add_argument('--sa-warmup-steps', type=int, default=10000, help='Delay adversarial training until this many environment steps have been collected.')
    sa_ddpg_cmd.add_argument('--sa-soc-new-threshold', type=float, default=0.5, help='O-scenario threshold for newly arrived vehicles during electhacker_O adversarial training.')
    sa_ddpg_cmd.add_argument('--sa-soc-rollout-threshold', type=float, default=0.3, help='O-scenario threshold for active rollout vehicles during electhacker_O adversarial training.')
    sa_ddpg_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_SA_DDPG_DIR)
    sa_ddpg_cmd.add_argument('--actor-model-name', type=str, default=None)
    sa_ddpg_cmd.add_argument('--bundle-path', type=str2path, default=None, help='Optional path for the actor+critic SA-DDPG bundle.')
    sa_ddpg_cmd.add_argument('--checkpoint-every', type=int, default=0, help='Save a continuous-training bundle every N episodes; 0 disables checkpointing.')
    sa_ddpg_cmd.add_argument('--checkpoint-dir', type=str2path, default=None, help='Directory for --checkpoint-every bundles.')
    sa_ddpg_cmd.add_argument('--checkpoint-prefix', type=str, default='sa_ddpg', help='Filename prefix for --checkpoint-every bundles.')
    sa_ddpg_cmd.set_defaults(func=command_train_sa_ddpg)


    ppo_lstm_cmd = subparsers.add_parser('train-online-ppo-lstm', help='Train the clean PPO-LSTM recurrent baseline for the ATLA ablation table.')
    add_runtime_args(ppo_lstm_cmd)
    ppo_lstm_cmd.add_argument('--outer-iters', type=int, default=90)
    ppo_lstm_cmd.add_argument('--phase-steps', type=int, default=2048)
    ppo_lstm_cmd.add_argument('--ppo-epochs', type=int, default=10)
    ppo_lstm_cmd.add_argument('--num-minibatches', type=int, default=32)
    ppo_lstm_cmd.add_argument('--gamma', type=float, default=0.99)
    ppo_lstm_cmd.add_argument('--gae-lambda', type=float, default=0.95)
    ppo_lstm_cmd.add_argument('--clip-eps', type=float, default=0.2)
    ppo_lstm_cmd.add_argument('--actor-lr', type=float, default=3e-4)
    ppo_lstm_cmd.add_argument('--critic-lr', type=float, default=3e-4)
    ppo_lstm_cmd.add_argument('--entropy-coeff', type=float, default=3e-4)
    ppo_lstm_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train_dense_safety')
    ppo_lstm_cmd.add_argument('--hidden-dim', type=int, default=128)
    ppo_lstm_cmd.add_argument('--lstm-dim', type=int, default=128)
    ppo_lstm_cmd.add_argument('--print-every', type=int, default=1)
    ppo_lstm_cmd.add_argument('--validation-every', type=int, default=0)
    ppo_lstm_cmd.add_argument('--init-bundle-path', type=str2path, default=None)
    ppo_lstm_cmd.add_argument('--iteration-offset', type=int, default=0)
    ppo_lstm_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_ONLINE_PPO_LSTM_DIR)
    ppo_lstm_cmd.add_argument('--bundle-path', type=str2path, default=None)
    ppo_lstm_cmd.add_argument('--history-path', type=str2path, default=None)
    ppo_lstm_cmd.set_defaults(func=command_train_online_ppo_lstm)

    eval_ppo_lstm_cmd = subparsers.add_parser('eval-online-ppo-lstm', help='Evaluate PPO-LSTM under no, random, and learned attacks.')
    add_runtime_args(eval_ppo_lstm_cmd)
    eval_ppo_lstm_cmd.add_argument('--bundle-path', type=str2path, default=None, help='Optional explicit PPO-LSTM bundle. Defaults to models/online_ppo_lstm/default_bundle.pt.')
    eval_ppo_lstm_cmd.add_argument('--epsilon', type=float, default=0.15)
    eval_ppo_lstm_cmd.add_argument('--attack-state-scope', type=str, default='local,all', help='local, all, or local,all.')
    eval_ppo_lstm_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train_dense_safety')
    eval_ppo_lstm_cmd.add_argument('--eval-adv-iters', type=int, default=50)
    eval_ppo_lstm_cmd.add_argument('--eval-adv-phase-steps', type=int, default=2048)
    eval_ppo_lstm_cmd.add_argument('--ppo-epochs', type=int, default=10)
    eval_ppo_lstm_cmd.add_argument('--num-minibatches', type=int, default=32)
    eval_ppo_lstm_cmd.add_argument('--gamma', type=float, default=0.99)
    eval_ppo_lstm_cmd.add_argument('--gae-lambda', type=float, default=0.95)
    eval_ppo_lstm_cmd.add_argument('--clip-eps', type=float, default=0.2)
    eval_ppo_lstm_cmd.add_argument('--adv-actor-lr', type=float, default=1e-3)
    eval_ppo_lstm_cmd.add_argument('--adv-critic-lr', type=float, default=1e-5)
    eval_ppo_lstm_cmd.add_argument('--adv-entropy-coeff', type=float, default=1e-4)
    eval_ppo_lstm_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_RESULTS_DIR / 'online_ppo_lstm')
    eval_ppo_lstm_cmd.add_argument('--eval-path', type=str2path, default=None)
    eval_ppo_lstm_cmd.set_defaults(func=command_eval_online_ppo_lstm)


    atla_cmd = subparsers.add_parser('train-atla', help='Train the single active ATLA policy: random-init PPO-LSTM + learned adversary + partial SA-Reg.')
    add_runtime_args(atla_cmd)
    atla_cmd.add_argument('--outer-iters', type=int, default=90)
    atla_cmd.add_argument('--phase-steps', type=int, default=2048)
    atla_cmd.add_argument('--ppo-epochs', type=int, default=10)
    atla_cmd.add_argument('--num-minibatches', type=int, default=32)
    atla_cmd.add_argument('--gamma', type=float, default=0.99)
    atla_cmd.add_argument('--gae-lambda', type=float, default=0.95)
    atla_cmd.add_argument('--clip-eps', type=float, default=0.2)
    atla_cmd.add_argument('--actor-lr', type=float, default=3e-4)
    atla_cmd.add_argument('--critic-lr', type=float, default=3e-4)
    atla_cmd.add_argument('--adv-actor-lr', type=float, default=1e-3)
    atla_cmd.add_argument('--adv-critic-lr', type=float, default=1e-5)
    atla_cmd.add_argument('--entropy-coeff', type=float, default=3e-4)
    atla_cmd.add_argument('--adv-entropy-coeff', type=float, default=1e-4)
    atla_cmd.add_argument('--sa-reg-weight', type=float, default=0.05)
    atla_cmd.add_argument('--sa-reg-steps', type=int, default=2)
    atla_cmd.add_argument('--epsilon', type=float, default=0.15)
    atla_cmd.add_argument('--attack-state-scope', choices=['local', 'all'], default='local')
    atla_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train_dense_safety')
    atla_cmd.add_argument('--hidden-dim', type=int, default=128)
    atla_cmd.add_argument('--lstm-dim', type=int, default=128)
    atla_cmd.add_argument('--adversary-hidden-dim', type=int, default=128)
    atla_cmd.add_argument('--print-every', type=int, default=1)
    atla_cmd.add_argument('--validation-every', type=int, default=5, help='Run deterministic clean/current-adversary validation every N outer iterations; 0 disables it.')
    atla_cmd.add_argument('--checkpoint-every', type=int, default=0, help='Save an ATLA bundle every N outer iterations; 0 disables it.')
    atla_cmd.add_argument('--checkpoint-dir', type=str2path, default=None)
    atla_cmd.add_argument('--checkpoint-prefix', type=str, default='atla')
    atla_cmd.add_argument('--init-bundle-path', type=str2path, default=None, help='Optional bundle to initialize/continue training from. Leave unset for the canonical random-init ATLA.')
    atla_cmd.add_argument('--iteration-offset', type=int, default=0)
    atla_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_ATLA_DIR)
    atla_cmd.add_argument('--bundle-path', type=str2path, default=DEFAULT_ATLA_BUNDLE)
    atla_cmd.add_argument('--history-path', type=str2path, default=None)
    atla_cmd.set_defaults(func=command_train_online_atla_ppo_lstm_sa)

    eval_atla_cmd = subparsers.add_parser('eval-atla', help='Evaluate the single active ATLA policy under clean/random/learned attacks.')
    add_runtime_args(eval_atla_cmd)
    eval_atla_cmd.add_argument('--bundle-path', type=str2path, default=DEFAULT_ATLA_BUNDLE)
    eval_atla_cmd.add_argument('--epsilon', type=float, default=0.15)
    eval_atla_cmd.add_argument('--attack-state-scope', type=str, default='local,all', help='local, all, or local,all.')
    eval_atla_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train_dense_safety')
    eval_atla_cmd.add_argument('--eval-adv-iters', type=int, default=50)
    eval_atla_cmd.add_argument('--eval-adv-phase-steps', type=int, default=2048)
    eval_atla_cmd.add_argument('--ppo-epochs', type=int, default=10)
    eval_atla_cmd.add_argument('--num-minibatches', type=int, default=32)
    eval_atla_cmd.add_argument('--gamma', type=float, default=0.99)
    eval_atla_cmd.add_argument('--gae-lambda', type=float, default=0.95)
    eval_atla_cmd.add_argument('--clip-eps', type=float, default=0.2)
    eval_atla_cmd.add_argument('--adv-actor-lr', type=float, default=1e-3)
    eval_atla_cmd.add_argument('--adv-critic-lr', type=float, default=1e-5)
    eval_atla_cmd.add_argument('--adv-entropy-coeff', type=float, default=1e-4)
    eval_atla_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_RESULTS_DIR / 'online_robust_policy' / 'atla')
    eval_atla_cmd.add_argument('--eval-path', type=str2path, default=None)
    eval_atla_cmd.set_defaults(func=command_eval_online_atla_ppo_lstm_sa)

    wocar_cmd = subparsers.add_parser('train-wocar', help='Train WocaR: baseline-warm-start clean-replay q-gap clipped robust actor-critic.')
    add_runtime_args(wocar_cmd)
    wocar_cmd.add_argument('--episodes', type=int, default=6)
    wocar_cmd.add_argument('--buffer-size', type=int, default=100000)
    wocar_cmd.add_argument('--batch-size', type=int, default=128)
    wocar_cmd.add_argument('--learning-starts', type=int, default=512)
    wocar_cmd.add_argument('--wocar-update-every', type=int, default=2, help='Run one WocaR replay update every N environment transitions after learning starts.')
    wocar_cmd.add_argument('--exploration-noise', type=float, default=0.2)
    wocar_cmd.add_argument('--gamma', type=float, default=0.9)
    wocar_cmd.add_argument('--tau', type=float, default=0.005)
    wocar_cmd.add_argument('--actor-lr', type=float, default=1e-4)
    wocar_cmd.add_argument('--critic-lr', type=float, default=3e-4)
    wocar_cmd.add_argument('--print-every', type=int, default=1)
    wocar_cmd.add_argument('--init-actor-path', type=str2path, default=None)
    wocar_cmd.add_argument('--resume-bundle-path', type=str2path, default=None)
    wocar_cmd.add_argument('--wocar-init-bundle-path', type=str2path, default=None, help='Optional baseline actor+critic bundle used for WocaR warm-start.')
    wocar_cmd.add_argument('--wocar-baseline-warmstart', action=argparse.BooleanOptionalAction, default=True, help='Warm-start WocaR from the baseline actor+critic bundle when no explicit init path is given.')
    wocar_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train_dense_safety')
    wocar_cmd.add_argument('--wocar-train-attacks', type=str, default='opposite_pgd,q_function', help='Comma-separated perturbation families used to build worst-case target and actor candidates.')
    wocar_cmd.add_argument('--wocar-epsilon', type=float, default=0.1, help='L-inf perturbation budget for worst-case candidates.')
    wocar_cmd.add_argument('--wocar-alpha', type=float, default=None, help='Optional inner perturbation step size override.')
    wocar_cmd.add_argument('--wocar-steps', type=int, default=None, help='Optional inner perturbation step-count override.')
    wocar_cmd.add_argument('--wocar-noise-std', type=float, default=0.0, help='Optional Gaussian noise added inside candidate search.')
    wocar_cmd.add_argument('--wocar-state-scope', choices=['local', 'global', 'all'], default='all', help='State dimensions perturbed while constructing worst-case candidates.')
    wocar_cmd.add_argument('--wocar-target-lambda', type=float, default=0.25, help='Blend weight for y_worst in the critic Bellman target.')
    wocar_cmd.add_argument('--wocar-actor-beta', type=float, default=0.6, help='Blend weight for the actor worst-case Q objective.')
    wocar_cmd.add_argument('--wocar-actor-q-weight', type=float, default=0.1, help='Extra normalized actor-loss weight for the q_function observation candidate.')
    wocar_cmd.add_argument('--wocar-actor-reg-weight', type=float, default=0.03, help='Weight for action consistency regularization between clean and adversarial observations.')
    wocar_cmd.add_argument('--wocar-actor-reg-weight-mode', choices=['uniform', 'q_gap'], default='q_gap', help='Per-state weight for the action consistency regularizer.')
    wocar_cmd.add_argument('--wocar-actor-reg-weight-clip', type=float, default=1.0, help='Upper clip for normalized q-gap reg weights. Use 0 to disable clipping.')
    wocar_cmd.add_argument('--wocar-validation-every', type=int, default=2, help='Run deterministic validation every N episodes and restore the best checkpoint. Use 0 to disable.')
    wocar_cmd.add_argument('--wocar-validation-attacks', type=str, default=None, help='Comma-separated attacks used for validation. Defaults to --wocar-train-attacks.')
    wocar_cmd.add_argument('--wocar-validation-baseline-bundle-path', type=str2path, default=None, help='Baseline actor+critic bundle used to score recovery. Defaults to the project baseline bundle.')
    wocar_cmd.add_argument('--wocar-validation-clean-drop-hard-cap', type=float, default=250.0, help='Validation checkpoints with clean drop above this reward cap are ineligible for best selection.')
    wocar_cmd.add_argument('--wocar-validation-clean-drop-weight', type=float, default=0.0, help='Optional within-cap clean-drop penalty for checkpoint scoring. The hard cap remains primary.')
    wocar_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_WOCAR_DIR)
    wocar_cmd.add_argument('--actor-model-name', type=str, default=None)
    wocar_cmd.add_argument('--bundle-path', type=str2path, default=None, help='Optional path for the actor+critic WocaR bundle.')
    wocar_cmd.set_defaults(func=command_train_wocar)

    atla_ddpg_cmd = subparsers.add_parser('train-atla-ddpg', help='Train the ATLA-DDPG branch without touching the PPO ATLA line.')
    add_runtime_args(atla_ddpg_cmd)
    atla_ddpg_cmd.add_argument('--episodes', type=int, default=120)
    atla_ddpg_cmd.add_argument('--buffer-size', type=int, default=100000)
    atla_ddpg_cmd.add_argument('--batch-size', type=int, default=256)
    atla_ddpg_cmd.add_argument('--learning-starts', type=int, default=2500)
    atla_ddpg_cmd.add_argument('--exploration-noise', type=float, default=1.0)
    atla_ddpg_cmd.add_argument('--gamma', type=float, default=0.9)
    atla_ddpg_cmd.add_argument('--tau', type=float, default=0.005)
    atla_ddpg_cmd.add_argument('--actor-lr', type=float, default=3e-4)
    atla_ddpg_cmd.add_argument('--critic-lr', type=float, default=3e-4)
    atla_ddpg_cmd.add_argument('--adv-actor-lr', type=float, default=1e-3, help='Learned observation adversary actor learning rate, matching PPO-ATLA naming.')
    atla_ddpg_cmd.add_argument('--adv-critic-lr', type=float, default=1e-5, help='Learned observation adversary value learning rate, matching PPO-ATLA naming.')
    atla_ddpg_cmd.add_argument('--ppo-epochs', type=int, default=4, help='PPO epochs for the learned adversary phase.')
    atla_ddpg_cmd.add_argument('--num-minibatches', type=int, default=16, help='Minibatches for learned adversary PPO updates.')
    atla_ddpg_cmd.add_argument('--adv-entropy-coeff', type=float, default=1e-4, help='Entropy coefficient for learned adversary PPO updates.')
    atla_ddpg_cmd.add_argument('--print-every', type=int, default=1)
    atla_ddpg_cmd.add_argument('--init-actor-path', type=str2path, default=None)
    atla_ddpg_cmd.add_argument('--resume-bundle-path', type=str2path, default=None)
    atla_ddpg_cmd.add_argument('--sa-init-bundle-path', type=str2path, default=None, help='Optional baseline actor+critic bundle for ATLA-DDPG warm-start.')
    atla_ddpg_cmd.add_argument('--sa-baseline-warmstart', action=argparse.BooleanOptionalAction, default=True, help='Warm-start ATLA-DDPG from the reward-profile-matched DDPG baseline.')
    atla_ddpg_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train_dense_safety')
    atla_ddpg_cmd.add_argument('--sa-train-attacks', type=str, default='', help='Deprecated/ignored for train-atla-ddpg: training now uses a learned ATLA adversary, not PGD/Q/FGSM cycling.')
    atla_ddpg_cmd.add_argument('--sa-epsilon', type=float, default=0.15, help='ATLA-style L-inf perturbation budget for the online state adversary.')
    atla_ddpg_cmd.add_argument('--sa-alpha', type=float, default=None)
    atla_ddpg_cmd.add_argument('--sa-steps', type=int, default=None)
    atla_ddpg_cmd.add_argument('--sa-objective', choices=['q_function', 'min_q', 'action'], default='q_function')
    atla_ddpg_cmd.add_argument('--sa-noise-std', type=float, default=0.0)
    atla_ddpg_cmd.add_argument('--sa-state-scope', choices=['local', 'global', 'all'], default='local')
    atla_ddpg_cmd.add_argument('--sa-actor-reg-weight', type=float, default=0.05, help='DDPG counterpart of ATLA partial SA regularization: clean-vs-attacked action consistency weight.')
    atla_ddpg_cmd.add_argument('--sa-mixed-update-attacks', action=argparse.BooleanOptionalAction, default=True)
    atla_ddpg_cmd.add_argument('--sa-anchor-actor-path', type=str2path, default=None)
    atla_ddpg_cmd.add_argument('--sa-anchor-reg-weight', type=float, default=0.1)
    atla_ddpg_cmd.add_argument('--sa-anchor-clean-weight', type=float, default=1.0)
    atla_ddpg_cmd.add_argument('--sa-clean-policy-weight', type=float, default=0.0)
    atla_ddpg_cmd.add_argument('--sa-risk-weight-scale', type=float, default=0.0)
    atla_ddpg_cmd.add_argument('--sa-risk-weight-max', type=float, default=3.0)
    atla_ddpg_cmd.add_argument('--sa-risk-target-soc', type=float, default=None)
    atla_ddpg_cmd.add_argument('--sa-validation-every', type=int, default=15)
    atla_ddpg_cmd.add_argument('--sa-validation-attacks', type=str, default=None)
    atla_ddpg_cmd.add_argument('--sa-validation-baseline-bundle-path', type=str2path, default=None)
    atla_ddpg_cmd.add_argument('--sa-validation-clean-drop-weight', type=float, default=0.3)
    atla_ddpg_cmd.add_argument('--sa-validation-clean-drop-budget', type=float, default=0.0)
    atla_ddpg_cmd.add_argument('--sa-validation-clean-drop-hard-cap', type=float, default=250.0)
    atla_ddpg_cmd.add_argument('--sa-validation-clean-exit-weight', type=float, default=0.0)
    atla_ddpg_cmd.add_argument('--sa-rollout-attack-prob-start', type=float, default=0.3)
    atla_ddpg_cmd.add_argument('--sa-rollout-attack-prob', type=float, default=1.0)
    atla_ddpg_cmd.add_argument('--sa-update-attack-prob-start', type=float, default=0.5)
    atla_ddpg_cmd.add_argument('--sa-update-attack-prob', type=float, default=1.0)
    atla_ddpg_cmd.add_argument('--sa-curriculum-steps', type=int, default=30000)
    atla_ddpg_cmd.add_argument('--sa-warmup-steps', type=int, default=10000)
    atla_ddpg_cmd.add_argument('--sa-soc-new-threshold', type=float, default=0.5)
    atla_ddpg_cmd.add_argument('--sa-soc-rollout-threshold', type=float, default=0.3)
    atla_ddpg_cmd.add_argument('--checkpoint-every', type=int, default=15)
    atla_ddpg_cmd.add_argument('--checkpoint-dir', type=str2path, default=DEFAULT_RESULTS_DIR / 'online_robust_policy' / 'atla_ddpg' / 'checkpoints')
    atla_ddpg_cmd.add_argument('--checkpoint-prefix', type=str, default='atla_ddpg')
    atla_ddpg_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_ATLA_DDPG_DIR)
    atla_ddpg_cmd.add_argument('--actor-model-name', type=str, default=None)
    atla_ddpg_cmd.add_argument('--bundle-path', type=str2path, default=DEFAULT_ATLA_DDPG_BUNDLE)
    atla_ddpg_cmd.add_argument('--history-path', type=str2path, default=None)
    atla_ddpg_cmd.set_defaults(func=command_train_atla_ddpg)

    eval_atla_ddpg_cmd = subparsers.add_parser('eval-atla-ddpg', help='Evaluate ATLA-DDPG against the clean DDPG baseline under short-horizon observation attacks.')
    add_runtime_args(eval_atla_ddpg_cmd)
    eval_atla_ddpg_cmd.add_argument('--sa-actor-path', type=str2path, default=None)
    eval_atla_ddpg_cmd.add_argument('--sa-bundle-path', type=str2path, default=DEFAULT_ATLA_DDPG_BUNDLE)
    eval_atla_ddpg_cmd.add_argument('--baseline-actor-path', type=str2path, default=None)
    eval_atla_ddpg_cmd.add_argument('--baseline-bundle-path', type=str2path, default=None)
    eval_atla_ddpg_cmd.add_argument('--eval-algorithms', type=str, default='opposite_pgd,opposite_fgsm,q_function')
    eval_atla_ddpg_cmd.add_argument('--eval-scenarios', type=str, default='O')
    eval_atla_ddpg_cmd.add_argument('--epsilons', type=str, default='0.1')
    eval_atla_ddpg_cmd.add_argument('--state-scope', choices=['local', 'global', 'all'], default='local')
    eval_atla_ddpg_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train_dense_safety')
    add_attack_tuning_args(eval_atla_ddpg_cmd)
    eval_atla_ddpg_cmd.add_argument('--save-dir', type=str2path, default=DEFAULT_RESULTS_DIR / 'online_robust_policy' / 'atla_ddpg' / 'short')
    eval_atla_ddpg_cmd.add_argument('--allow-cross-attack-eval', action=argparse.BooleanOptionalAction, default=True)
    eval_atla_ddpg_cmd.set_defaults(func=command_evaluate_sa_ddpg)

    eval_sa_ddpg_cmd = subparsers.add_parser('evaluate-sa-ddpg', help='Evaluate an online SA-DDPG policy against the baseline with the same attack families and output a paper-style comparison table.')
    add_runtime_args(eval_sa_ddpg_cmd)
    eval_sa_ddpg_cmd.add_argument('--sa-actor-path', type=str2path, default=None, help='SA-DDPG actor path. If omitted, --sa-bundle-path is used for both actor and critic.')
    eval_sa_ddpg_cmd.add_argument('--sa-bundle-path', type=str2path, required=True, help='SA-DDPG actor+critic bundle for white-box q_function evaluation.')
    eval_sa_ddpg_cmd.add_argument('--baseline-actor-path', type=str2path, default=None, help='Baseline actor path. Defaults to the project baseline actor.')
    eval_sa_ddpg_cmd.add_argument('--baseline-bundle-path', type=str2path, default=None, help='Baseline actor+critic bundle. Defaults to the project baseline bundle.')
    eval_sa_ddpg_cmd.add_argument('--eval-algorithms', type=str, default='opposite_pgd,q_function', help='Comma-separated attacks used to compare baseline and SA-DDPG.')
    eval_sa_ddpg_cmd.add_argument('--epsilons', type=str, default='0.1', help='Comma-separated attack epsilons.')
    eval_sa_ddpg_cmd.add_argument('--state-scope', choices=['local', 'global', 'all'], default='all', help='State dimensions attacked during evaluation.')
    eval_sa_ddpg_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    add_attack_tuning_args(eval_sa_ddpg_cmd)
    eval_sa_ddpg_cmd.add_argument('--save-dir', type=str2path, default=DEFAULT_RESULTS_DIR / 'sa_ddpg')
    eval_sa_ddpg_cmd.add_argument('--allow-cross-attack-eval', action=argparse.BooleanOptionalAction, default=False, help='Allow evaluation attacks that were not listed in the bundle training metadata.')
    eval_sa_ddpg_cmd.set_defaults(func=command_evaluate_sa_ddpg)


    collect_clean_cmd = subparsers.add_parser('collect-clean', help='Collect and save clean Dnormal rollouts.')
    add_runtime_args(collect_clean_cmd)
    add_actor_args(collect_clean_cmd)
    add_clean_dataset_args(collect_clean_cmd)
    collect_clean_cmd.add_argument('--episodes', type=int, default=1)
    collect_clean_cmd.add_argument('--max-samples', type=int, default=None)
    collect_clean_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    collect_clean_cmd.add_argument('--save-path', type=str2path, default=None)
    collect_clean_cmd.set_defaults(func=command_collect_clean)

    eval_rollout = subparsers.add_parser('evaluate-rollout', help='Evaluate clean/attack/defended rollout performance.')
    add_runtime_args(eval_rollout)
    add_actor_args(eval_rollout)
    add_attack_args(eval_rollout)
    eval_rollout.add_argument('--dae-path', type=str2path, default=None)
    add_detector_runtime_args(eval_rollout)
    eval_rollout.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    eval_rollout.add_argument('--exploration-noise', type=float, default=0.0)
    eval_rollout.add_argument('--save-dir', type=str2path, default=None)
    eval_rollout.set_defaults(func=command_evaluate_rollout)

    eval_actions = subparsers.add_parser('evaluate-actions', help='Evaluate action deviation on offline pair data.')
    add_runtime_args(eval_actions)
    add_actor_args(eval_actions)
    add_attack_args(eval_actions)
    eval_actions.add_argument('--pair-dir', type=str2path, default=DEFAULT_PAIR_DIR)
    eval_actions.add_argument('--pair-path', type=str2path, default=None)
    eval_actions.add_argument('--dae-path', type=str2path, default=None)
    eval_actions.add_argument('--save-dir', type=str2path, default=None)
    eval_actions.set_defaults(func=command_evaluate_actions)

    temporal_shield_cmd = subparsers.add_parser('train-offline-dae-det-temporal-shield', help='Calibrate and save the GRU-VAE DAE + posterior-det temporal shield bundle.')
    add_runtime_args(temporal_shield_cmd)
    add_actor_args(temporal_shield_cmd)
    temporal_shield_cmd.add_argument('--state-scope', choices=['local', 'all'], default='local')
    temporal_shield_cmd.add_argument('--dae-path', type=str2path, default=None, help='Defaults to the GRU-VAE q+pgd bundle for the chosen scope.')
    temporal_shield_cmd.add_argument('--detector-path', type=str2path, default=None, help='Defaults to the posterior detector bundle for the chosen scope.')
    temporal_shield_cmd.add_argument('--detector-threshold', type=float, default=None, help='Override the detector threshold stored in the detector artifact.')
    temporal_shield_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    temporal_shield_cmd.add_argument('--calibration-quantile', type=float, default=0.99)
    temporal_shield_cmd.add_argument('--min-tau-soc', type=float, default=0.02)
    temporal_shield_cmd.add_argument('--min-tau-time', type=float, default=0.005)
    temporal_shield_cmd.add_argument('--min-tau-cost', type=float, default=0.02)
    temporal_shield_cmd.add_argument('--max-tau-soc', type=float, default=0.08)
    temporal_shield_cmd.add_argument('--max-tau-time', type=float, default=0.03)
    temporal_shield_cmd.add_argument('--max-tau-cost', type=float, default=0.08)
    temporal_shield_cmd.add_argument('--clean-drop-limit', type=float, default=50.0)
    temporal_shield_cmd.add_argument('--tune-with-attacks', action=argparse.BooleanOptionalAction, default=True)
    temporal_shield_cmd.add_argument('--tau-soc-scales', type=str, default='0.75,1.0,1.25')
    temporal_shield_cmd.add_argument('--tau-time-scales', type=str, default='1.0,1.5')
    temporal_shield_cmd.add_argument('--tau-cost-scales', type=str, default='0.75,1.0,1.25')
    temporal_shield_cmd.add_argument('--attack-bundle-path', type=str2path, default=None, help='Bundle path used to load critic weights for q-function validation attacks. Defaults to the baseline bundle.')
    temporal_shield_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_OFFLINE_DAE_DET_TEMPORAL_SHIELD_DIR)
    temporal_shield_cmd.add_argument('--output-path', type=str2path, default=None)
    temporal_shield_cmd.set_defaults(actor_path=Path(DEFAULT_BASELINE_ACTOR_PATH))
    temporal_shield_cmd.set_defaults(func=command_train_offline_dae_det_temporal_shield)

    eval_temporal_shield_cmd = subparsers.add_parser('eval-offline-dae-det-temporal-shield', help='Evaluate the GRU-VAE DAE + posterior-det temporal shield line against q/pgd/learned attacks.')
    add_runtime_args(eval_temporal_shield_cmd)
    add_actor_args(eval_temporal_shield_cmd)
    eval_temporal_shield_cmd.add_argument('--state-scope', choices=['local', 'all'], default='local')
    eval_temporal_shield_cmd.add_argument('--shield-dir', type=str2path, default=DEFAULT_OFFLINE_DAE_DET_TEMPORAL_SHIELD_DIR)
    eval_temporal_shield_cmd.add_argument('--shield-path', type=str2path, default=None)
    eval_temporal_shield_cmd.add_argument('--dae-path', type=str2path, default=None)
    eval_temporal_shield_cmd.add_argument('--detector-path', type=str2path, default=None)
    eval_temporal_shield_cmd.add_argument('--detector-threshold', type=float, default=None)
    eval_temporal_shield_cmd.add_argument('--eval-algorithms', type=str, default='opposite_pgd,q_function')
    eval_temporal_shield_cmd.add_argument('--epsilon-q-pgd', type=float, default=0.1)
    eval_temporal_shield_cmd.add_argument('--epsilon-learned', type=float, default=0.15)
    eval_temporal_shield_cmd.add_argument('--scenario', choices=ATTACK_SCENARIOS, default='O')
    eval_temporal_shield_cmd.add_argument('--attack-bundle-path', type=str2path, default=Path(DEFAULT_BASELINE_BUNDLE_PATH))
    add_attack_tuning_args(eval_temporal_shield_cmd)
    eval_temporal_shield_cmd.add_argument('--reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    eval_temporal_shield_cmd.add_argument('--learned-adv-iters', type=int, default=200)
    eval_temporal_shield_cmd.add_argument('--learned-adv-phase-steps', type=int, default=2048)
    eval_temporal_shield_cmd.add_argument('--learned-adv-ppo-epochs', type=int, default=10)
    eval_temporal_shield_cmd.add_argument('--learned-adv-num-minibatches', type=int, default=32)
    eval_temporal_shield_cmd.add_argument('--gamma', type=float, default=0.99)
    eval_temporal_shield_cmd.add_argument('--gae-lambda', type=float, default=0.95)
    eval_temporal_shield_cmd.add_argument('--clip-eps', type=float, default=0.2)
    eval_temporal_shield_cmd.add_argument('--adv-actor-lr', type=float, default=1e-3)
    eval_temporal_shield_cmd.add_argument('--adv-critic-lr', type=float, default=1e-5)
    eval_temporal_shield_cmd.add_argument('--adv-entropy-coeff', type=float, default=1e-4)
    eval_temporal_shield_cmd.add_argument('--adversary-hidden-dim', type=int, default=128)
    eval_temporal_shield_cmd.add_argument('--print-every', type=int, default=10)
    eval_temporal_shield_cmd.add_argument('--output-dir', type=str2path, default=DEFAULT_OFFLINE_DAE_DET_TEMPORAL_SHIELD_RESULTS_DIR)
    eval_temporal_shield_cmd.add_argument('--output-path', type=str2path, default=None)
    eval_temporal_shield_cmd.set_defaults(actor_path=Path(DEFAULT_BASELINE_ACTOR_PATH))
    eval_temporal_shield_cmd.set_defaults(func=command_eval_offline_dae_det_temporal_shield)

    unified_cmd = subparsers.add_parser('run-unified', help='Run the unified paper-style Dnormal -> Dadv -> DAE + detector pipeline.')
    add_unified_run_args(unified_cmd)
    unified_cmd.set_defaults(func=command_run_unified)


    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


# === detector paper-style CLI helpers ===
DETECTOR_FEATURE_MODES = ('posterior', 'sequence')
POSTERIOR_LABEL_MODES = ('benefit', 'attack')


def canonical_detector_feature_mode(value: str | None) -> str:
    token = str(value or 'posterior').strip().lower().replace('-', '_')
    if token in {'posterior', 'posterior_mlp', 'post'}:
        return 'posterior'
    if token in {'sequence', 'seq', 'gruvae', 'gru_vae'}:
        return 'sequence'
    raise ValueError(f'Unsupported detector feature mode: {value}')


def canonical_posterior_label_mode(value: str | None) -> str:
    token = str(value or 'benefit').strip().lower().replace('-', '_')
    if token in {'benefit', 'repair_benefit', 'utility', 'gain'}:
        return 'benefit'
    if token in {'attack', 'attack_clean', 'attacked', 'attack_label'}:
        return 'attack'
    raise ValueError(f'Unsupported posterior label mode: {value}')


def detector_feature_suffix(detector_feature_mode: str, posterior_label_mode: str | None = 'benefit') -> str:
    token = canonical_detector_feature_mode(detector_feature_mode)
    if token != 'posterior':
        return '_seq'
    label_mode = canonical_posterior_label_mode(posterior_label_mode)
    return '_post' if label_mode == 'benefit' else f'_post_{label_mode}'


def detector_dataset_mode_for_feature_mode(detector_feature_mode: str) -> str:
    return 'posterior' if canonical_detector_feature_mode(detector_feature_mode) == 'posterior' else 'pre'


def validate_detector_dataset_mode(
    dataset: DetectorDatasetBundle,
    detector_feature_mode: str,
    path: Path | str,
    *,
    posterior_label_mode: str | None = 'benefit',
) -> None:
    expected = detector_dataset_mode_for_feature_mode(detector_feature_mode)
    actual = str((dataset.metadata or {}).get('detector_mode', 'pre')).strip().lower()
    if actual != expected:
        raise ValueError(f'Detector dataset mode mismatch for {path}: expected {expected!r}, got {actual!r}.')
    if expected == 'posterior':
        expected_label = canonical_posterior_label_mode(posterior_label_mode)
        actual_label = canonical_posterior_label_mode((dataset.metadata or {}).get('posterior_label_mode', 'benefit'))
        if actual_label != expected_label:
            raise ValueError(f'Posterior detector dataset label mode mismatch for {path}: expected {expected_label!r}, got {actual_label!r}.')


def add_detector_runtime_args(parser: argparse.ArgumentParser, *, prefix: str = '', default_feature_mode: str = 'sequence') -> None:
    parser.add_argument(f'--{prefix}detector-path', type=str2path, default=None, help='Explicit detector artifact path.')
    parser.add_argument(f'--{prefix}detector-threshold', type=float, default=None, help='Override detector anomaly threshold (Canomaly).')
    parser.add_argument(f'--{prefix}detector-feature-mode', choices=DETECTOR_FEATURE_MODES, default=canonical_detector_feature_mode(default_feature_mode), help='Detector backend. `posterior` means DAE-first repair-benefit filter; `sequence` keeps the old pre-DAE GRU-VAE anomaly detector.')


def add_detector_train_args(parser: argparse.ArgumentParser, *, prefix: str = '', default_feature_mode: str = 'sequence') -> None:
    parser.add_argument(f'--{prefix}detector-epochs', type=int, default=30, help='Detector training epochs.')
    parser.add_argument(f'--{prefix}detector-batch-size', type=int, default=256, help='Detector training batch size.')
    parser.add_argument(f'--{prefix}detector-lr', type=float, default=1e-3, help='Detector learning rate.')
    parser.add_argument(f'--{prefix}detector-hidden-dim', type=int, default=128, help='Detector GRU hidden dimension.')
    parser.add_argument(f'--{prefix}detector-latent-dim', type=int, default=64, help='Detector latent dimension.')
    parser.add_argument(f'--{prefix}detector-num-layers', type=int, default=1, help='Detector GRU layer count.')
    parser.add_argument(f'--{prefix}detector-beta-kl', type=float, default=1e-3, help='Detector KL weight.')
    parser.add_argument(f'--{prefix}detector-seq-len', type=int, default=8, help='Detector sequence window size.')
    parser.add_argument(f'--{prefix}detector-val-ratio', type=float, default=0.2, help='Detector validation split ratio.')


def resolve_detector(args, device: torch.device, *, prefix: str = '', default_feature_mode: str = 'sequence', default_path: Path | None = None, expected_metadata: dict | None = None, required_metadata: dict | None = None):
    path_attr = prefixed_attr(prefix, 'detector_path')
    threshold_attr = prefixed_attr(prefix, 'detector_threshold')
    feature_mode_attr = prefixed_attr(prefix, 'detector_feature_mode')
    detector_path = getattr(args, path_attr, None) or default_path
    detector_threshold = getattr(args, threshold_attr, None)
    if detector_path is None and detector_threshold is None:
        return None
    model = None
    threshold = None
    metadata = {}
    feature_mode = canonical_detector_feature_mode(getattr(args, feature_mode_attr, default_feature_mode))
    if detector_path is not None:
        artifact = load_detector(detector_path, device)
        model = artifact.model
        threshold = float(artifact.threshold)
        metadata = dict(artifact.metadata or {})
        if expected_metadata is not None or required_metadata is not None:
            _validate_metadata(metadata, artifact_kind='detector artifact', artifact_path=detector_path, expected=expected_metadata, required_values=required_metadata)
        feature_mode = canonical_detector_feature_mode(metadata.get('detector_feature_mode', feature_mode))
    if detector_threshold is not None:
        threshold = float(detector_threshold)
    return {'model': model, 'threshold': threshold, 'metadata': metadata, 'feature_mode': feature_mode}


def add_unified_run_args(parser: argparse.ArgumentParser) -> None:
    add_runtime_args(parser)
    add_actor_args(parser)
    add_clean_dataset_args(parser)
    parser.add_argument('--train-algorithms', type=str, default='opposite_pgd,q_function', help='Comma-separated attacks used to build unified DAE training pairs.')
    parser.add_argument('--eval-algorithms', type=str, default='opposite_pgd,q_function', help='Comma-separated attacks evaluated with the unified DAE/detector.')
    parser.add_argument('--eval-scenarios', type=str, default='O', help='ElectHacker scenarios for eval-only attacks.')
    parser.add_argument('--epsilons', type=str, default='0.1', help='Comma-separated epsilon values.')
    parser.add_argument('--profile-tag', type=str, default=None, help='Optional stable tag for unified artifacts.')
    parser.add_argument('--state-scope', choices=['local', 'all'], default='all', help='State dimensions attacked and defended together for the paper mainline.')
    parser.add_argument('--attack-bundle-path', type=str2path, default=None, help='Optional actor+critic bundle for q_function attack.')
    parser.add_argument('--alpha', type=float, default=None, help='Attack step size; defaults per attack family.')
    parser.add_argument('--iters', type=int, default=None, help='Attack iterations; defaults per attack family.')
    parser.add_argument('--price-threshold', type=float, default=400.0)
    parser.add_argument('--soc-new-threshold', type=float, default=0.5)
    parser.add_argument('--soc-rollout-threshold', type=float, default=0.3)
    parser.add_argument('--even-station-target', type=float, default=1.0)
    parser.add_argument('--odd-station-target', type=float, default=-0.5)
    parser.add_argument('--pair-dir', type=str2path, default=DEFAULT_PAIR_DIR)
    parser.add_argument('--pair-path', type=str2path, default=None)
    parser.add_argument('--dae-dir', type=str2path, default=DEFAULT_DAE_DIR)
    parser.add_argument('--dae-path', type=str2path, default=None)
    parser.add_argument('--detector-data-dir', type=str2path, default=DEFAULT_DETECTOR_DATA_DIR)
    parser.add_argument('--detector-dataset-path', type=str2path, default=None)
    parser.add_argument('--detector-dir', type=str2path, default=DEFAULT_DETECTOR_DIR)
    parser.add_argument('--detector-path', type=str2path, default=None)
    parser.add_argument('--detector-feature-mode', choices=DETECTOR_FEATURE_MODES, default='posterior', help='Unified detector backend. `posterior` is the default DAE-first accept/reject filter; `sequence` keeps the old pre-DAE anomaly detector as an ablation.')
    parser.add_argument('--pair-source', choices=['collect', 'load', 'load_or_collect'], default='load_or_collect')
    parser.add_argument('--dae-source', choices=['train', 'load', 'load_or_train'], default='load_or_train')
    parser.add_argument('--detector-data-source', choices=['collect', 'load', 'load_or_collect'], default='load_or_collect')
    parser.add_argument('--detector-source', choices=['tune', 'load', 'load_or_tune'], default='load_or_tune')
    parser.add_argument('--save-pairs', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-daes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-detector-data', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-detectors', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--data-only', action=argparse.BooleanOptionalAction, default=False, help='Only build/save formal Dnormal, unified pair, and detector datasets; skip DAE/detector training and rollout evaluation.')
    parser.add_argument('--collect-reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    parser.add_argument('--rollout-reward-profile', choices=list(PROFILE_MAP.keys()), default='train')
    parser.add_argument('--collect-max-samples', type=int, default=2048)
    parser.add_argument('--collect-episodes', type=int, default=1)
    parser.add_argument('--dae-epochs', type=int, default=25)
    parser.add_argument('--dae-batch-size', type=int, default=128)
    parser.add_argument('--dae-lr', type=float, default=1e-3)
    parser.add_argument('--lambda-state', type=float, default=1.0)
    parser.add_argument('--lambda-identity', type=float, default=2.0)
    parser.add_argument('--dae-checkpoint-metric', choices=['rollout', 'loss'], default='rollout', help='Metric used to choose the best DAE checkpoint during training.')
    parser.add_argument('--dae-val-every', type=int, default=5, help='Evaluate the DAE checkpoint metric every N epochs.')
    parser.add_argument('--dae-checkpoint-clean-penalty', type=float, default=1.0, help='Penalty weight for clean-side degradation in DAE checkpoint selection.')
    parser.add_argument('--posterior-benefit-margin', type=float, default=0.0, help='Optional dead-zone around zero benefit when labeling posterior-det training samples.')
    parser.add_argument('--posterior-benefit-action-weight', type=float, default=1.0, help='Weight on clean-action improvement when labeling posterior-det samples.')
    parser.add_argument('--posterior-benefit-state-weight', type=float, default=1.0, help='Weight on clean-state improvement when labeling posterior-det samples.')
    parser.add_argument('--posterior-label-mode', choices=POSTERIOR_LABEL_MODES, default='benefit', help='Posterior detector target: repair-benefit routing label or attack/clean label with the same post-DAE features.')
    parser.add_argument('--posterior-use-benefit-weights', action=argparse.BooleanOptionalAction, default=True, help='Use benefit-magnitude sample weights for benefit-label posterior detector training.')
    add_dae_model_args(parser)
    add_detector_selection_args(parser, default_grid_size=31)
    add_detector_train_args(parser)
    parser.add_argument('--exploration-noise', type=float, default=0.0)
    parser.add_argument('--output-dir', type=str2path, default=DEFAULT_UNIFIED_DIR)



if __name__ == '__main__':
    main()
