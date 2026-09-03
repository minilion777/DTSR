"""项目核心模块。

这里统一收纳：
- 通用工具函数；
- data.csv / signals.json 读取；
- 环境定义；
- 策略网络、价值网络与基础 DDPG 代理。

这样可以把原来散落在 11 个脚本中的重复逻辑集中到一处。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

DEFAULT_DATA_PATH = DATA_DIR / 'data.csv'
DEFAULT_SIGNALS_PATH = DATA_DIR / 'signals.json'
DEFAULT_BASELINE_ACTOR_PATH = MODELS_DIR / 'baseline' / 'actor_baseline_ep50_seed42.pt'
DEFAULT_BASELINE_BUNDLE_PATH = MODELS_DIR / 'baseline' / 'baseline_bundle_ep50_seed42.pt'
LEGACY_BASELINE_ACTOR_PATH = MODELS_DIR / 'baseline' / 'actor_baseline_best.pt'
LEGACY_BASELINE_BUNDLE_PATH = MODELS_DIR / 'baseline' / 'baseline_bundle_best.pt'
DEFAULT_RESULTS_DIR = PROJECT_ROOT / 'results'
DEFAULT_EPSILON_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

ATTACK_ALGORITHMS = (
    'electhacker',
    'opposite_pgd',
    'opposite_fgsm',
    'q_function',
    'critic_v',
    'action',
    'advpolicy',
    'pgd',
    'fgsm',
)
ATTACK_SCENARIOS = ('C', 'F', 'O')
POLICY_MODES = ('baseline',)


@dataclass(frozen=True)
class RewardProfile:
    """把训练/评估里的 SOC 约束显式参数化。

    这里额外兼容 DDPG-DEFENSE.py 里的动作幅度惩罚：
    当 |a| 超过阈值时，额外扣减奖励。
    """

    name: str
    exit_target_min: float
    exit_target_max: float
    running_soc_min: float
    running_soc_max: float = 1.0
    reward_soc_weight: float = 2.0
    action_penalty_threshold: float | None = None
    action_penalty_scale: float = 0.0
    dense_safety_penalty_weight: float = 0.0
    dense_safety_target_soc: float | None = None


TRAIN_PROFILE = RewardProfile('train', 0.9, 1.0, 0.15)
DENSE_SAFETY_PROFILE = RewardProfile(
    'train_dense_safety',
    0.9,
    1.0,
    0.15,
    dense_safety_penalty_weight=1.0,
)
PROFILE_MAP = {
    'train': TRAIN_PROFILE,
    'train_dense_safety': DENSE_SAFETY_PROFILE,
}


RESULT_KEY_RENAME_MAP = {
    'ep_r1_cost_sum': 'ep_r1',
    'ep_r2_exit_penalty_sum': 'ep_r2',
    'ep_r3_running_penalty_sum': 'ep_r3',
    'ep_r4_dense_safety_penalty_sum': 'ep_r4_dense',
    'gini_cost': 'gini',
    'mean_final_soc': 'mean_fin_soc',
    'std_final_soc': 'std_finl_soc',
    'final_soc_count': 'fin_soc_count',
    'exit_violation_count': 'exit_vio',
    'running_violation_count': 'run_vio',
    'done_count': 'done_cnt',
    'cost_count': 'cost_cnt',
}
RESULT_FLOAT_DIGITS = 2


def normalize_result_object(obj: object, *, rename_keys: bool = True, digits: int = RESULT_FLOAT_DIGITS) -> object:
    """递归重命名结果字段，并把所有数值保留固定小数位。"""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {
            (RESULT_KEY_RENAME_MAP.get(str(k), str(k)) if rename_keys else str(k)): normalize_result_object(v, rename_keys=rename_keys, digits=digits)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [normalize_result_object(v, rename_keys=rename_keys, digits=digits) for v in obj]
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return round(float(obj), digits)
    return obj


def normalize_result_frame(df: pd.DataFrame, *, rename_keys: bool = True, digits: int = RESULT_FLOAT_DIGITS) -> pd.DataFrame:
    """把结果表统一成目标字段名，并将数值保留固定小数位。"""
    out = df.copy()
    if rename_keys:
        out = out.rename(columns=RESULT_KEY_RENAME_MAP)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        out[col] = out[col].astype(float).round(digits)
    return out


@dataclass(frozen=True)
class AttackDefaults:
    """不同攻击器的默认超参数。"""

    epsilon: float
    alpha: float
    iters: int


ATTACK_DEFAULTS = {
    'electhacker': AttackDefaults(0.1, 0.005, 100),
    'opposite_pgd': AttackDefaults(0.1, 0.01, 10),
    'opposite_fgsm': AttackDefaults(0.1, 0.1, 1),
    'q_function': AttackDefaults(0.1, 0.01, 10),
    'critic_v': AttackDefaults(0.1, 0.01, 10),
    'action': AttackDefaults(0.1, 0.01, 10),
    'advpolicy': AttackDefaults(0.1, 0.01, 1),
    # Backward-compatible aliases.
    'pgd': AttackDefaults(0.1, 0.01, 10),
    'fgsm': AttackDefaults(0.1, 0.1, 1),
}


def canonical_attack_algorithm(algorithm: str) -> str:
    """Normalize user-facing attack names to the paper-aligned family names."""
    token = str(algorithm).strip().lower()
    alias_map = {
        'pgd': 'opposite_pgd',
        'fgsm': 'opposite_fgsm',
        'opposite_pgd': 'opposite_pgd',
        'opposite_fgsm': 'opposite_fgsm',
        'q_function': 'q_function',
        'q-function': 'q_function',
        'q_function_attack': 'q_function',
        'q-function-attack': 'q_function',
        'critic': 'critic_v',
        'critic_v': 'critic_v',
        'critic-v': 'critic_v',
        'value': 'critic_v',
        'value_attack': 'critic_v',
        'value-attack': 'critic_v',
        'action': 'action',
        'max_action_diff': 'action',
        'mad': 'action',
        'advpolicy': 'advpolicy',
        'adv_policy': 'advpolicy',
        'adv-policy': 'advpolicy',
        'optimal': 'advpolicy',
        'optimal_attack': 'advpolicy',
        'optimal-attack': 'advpolicy',
        'electhacker': 'electhacker',
    }
    if token not in alias_map:
        raise ValueError(f'Unknown attack algorithm: {algorithm}')
    return alias_map[token]


@dataclass
class Signals:
    """固定外生信号。"""

    wt: np.ndarray
    pv: np.ndarray
    load: np.ndarray
    price: np.ndarray
    norm_price: np.ndarray
    min_price: float
    max_price: float


@dataclass
class QueueItem:
    """队列中的一辆车。"""

    obs: np.ndarray
    station: int


@dataclass
class Transition:
    """单步转移样本。"""

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    station: int
    step_cost: float
    exit_penalty: float
    running_penalty: float
    dense_safety_penalty: float
    exit_violation_count: int
    running_violation_count: int
    final_soc: float


@dataclass
class EpisodeMetrics:
    """一整天 rollout 的累计指标。"""

    ep_reward: float = 0.0
    ep_r1_cost_sum: float = 0.0
    ep_r2_exit_penalty_sum: float = 0.0
    ep_r3_running_penalty_sum: float = 0.0
    ep_r4_dense_safety_penalty_sum: float = 0.0
    exit_violation_count: int = 0
    running_violation_count: int = 0
    total_transitions: int = 0
    done_count: int = 0
    costlist: list[float] = field(default_factory=list)
    final_soc_list: list[float] = field(default_factory=list)
    powercurve: list[float] = field(default_factory=list)
    powerlist: list[list[float]] = field(default_factory=lambda: [[] for _ in range(9)])


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """保证目录存在。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def split_csv_strings(text: str) -> list[str]:
    """把逗号分隔字符串转成列表。"""
    return [x.strip() for x in text.split(',') if x.strip()]



def split_csv_floats(text: str) -> list[float]:
    """把逗号分隔浮点字符串转成列表。"""
    return [float(x.strip()) for x in text.split(',') if x.strip()]



def min_max_normalization(data: Sequence[float], min_val: float = 0.0, max_val: float = 1.0) -> list[float]:
    """把数列线性归一化到 [min_val, max_val]。"""
    data = list(data)
    min_data = min(data)
    max_data = max(data)
    if math.isclose(min_data, max_data):
        return [float(min_val) for _ in data]
    return [
        float(min_val + (x - min_data) * (max_val - min_val) / (max_data - min_data))
        for x in data
    ]



def min_max_denormalization(
    normalized_data: float | Sequence[float],
    original_min: float,
    original_max: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float | list[float]:
    """把归一化值恢复到原始量纲。"""
    def _restore(x: float) -> float:
        if math.isclose(max_val, min_val):
            return float(original_min)
        return float(original_min + (x - min_val) * (original_max - original_min) / (max_val - min_val))

    if isinstance(normalized_data, (list, tuple, np.ndarray)):
        return [_restore(float(x)) for x in normalized_data]
    return _restore(float(normalized_data))



def normalize_scalar(value: float, min_value: float, max_value: float) -> float:
    """对单个标量做归一化。"""
    if math.isclose(min_value, max_value):
        return 0.0
    return float((value - min_value) / (max_value - min_value))



def to_numpy_1d(x) -> np.ndarray:
    """把输入整理成 float32 一维数组。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def resolve_max_duration_of_stay(arrivals: pd.DataFrame | None, default: float = 12.0) -> float:
    """Infer the maximum duration-of-stay in 15-minute slots from arrivals."""
    if arrivals is None or 'Duration_of_stay' not in arrivals.columns or len(arrivals) == 0:
        return float(default)
    values = pd.to_numeric(arrivals['Duration_of_stay'], errors='coerce').to_numpy(dtype=np.float64, copy=False)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(max(float(default), float(values.max(initial=float(default)))))



def gini(values: Sequence[float]) -> float:
    """计算基尼系数。"""
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size == 0 or np.allclose(arr, 0.0):
        return 0.0
    min_val = float(arr.min())
    if min_val < 0:
        arr = arr - min_val
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    num = float(np.sum((2 * idx - n - 1) * arr))
    den = float(n * np.sum(arr))
    return 0.0 if math.isclose(den, 0.0) else num / den



def json_dump(data: object, path: str | Path, *, normalize_numbers: bool = False, rename_keys: bool = True) -> Path:
    """把对象保存成 UTF-8 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = normalize_result_object(data, rename_keys=rename_keys) if normalize_numbers else data
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path



def load_arrivals(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """读取 data.csv，并清理掉无用索引列。"""
    df = pd.read_csv(path)
    drop_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    required = ['Arrive_time', 'Duration_of_stay', 'Station']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'data.csv 缺少必要列: {missing}')
    df = df[required].copy()
    for col in required:
        df[col] = df[col].astype(int)
    return df.sort_values(['Arrive_time', 'Station']).reset_index(drop=True)



def generate_synthetic_arrivals(seed: int = 42, num_sessions: int = 344) -> pd.DataFrame:
    """当 data.csv 缺失时，生成一份兼容格式的合成数据。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            'Arrive_time': rng.integers(1, 86, size=num_sessions),
            'Duration_of_stay': rng.integers(4, 13, size=num_sessions),
            'Station': rng.integers(0, 9, size=num_sessions),
        }
    )
    return df.sort_values(['Arrive_time', 'Station']).reset_index(drop=True)



def ensure_arrivals(path: str | Path = DEFAULT_DATA_PATH, seed: int = 42) -> Path:
    """保证 data.csv 存在；如果不存在则自动生成。"""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_arrivals(seed=seed).to_csv(path, index=False)
    return path



def load_signals(path: str | Path = DEFAULT_SIGNALS_PATH) -> Signals:
    """读取原始项目固定的风、光、负荷、电价信号。"""
    with Path(path).open('r', encoding='utf-8') as f:
        raw = json.load(f)
    wt = np.asarray(raw['WT'], dtype=np.float32) / 100.0
    pv = np.asarray(raw['PV'], dtype=np.float32) / 100.0
    load = np.asarray(raw['L'], dtype=np.float32) / 100.0
    price = np.asarray(raw['price'], dtype=np.float32)
    norm_price = np.asarray(min_max_normalization(price.tolist(), 0.0, 1.0), dtype=np.float32)
    return Signals(wt=wt, pv=pv, load=load, price=price, norm_price=norm_price, min_price=float(price.min()), max_price=float(price.max()))


class ChargingEnv:
    """原始项目 Environment_1 的模块化版本。"""

    obs_dim = 11
    action_dim = 1
    horizon = 96

    def __init__(
        self,
        signals_path: str | Path = DEFAULT_SIGNALS_PATH,
        reward_profile: RewardProfile = TRAIN_PROFILE,
        station_count: int = 9,
        battery_capacity: float = 0.04992,
        max_power: float = 0.07,
        slice_hours: float = 0.25,
        initial_cost_norm: float = 0.2,
        initial_soc: float = 0.0,
    ) -> None:
        self.signals = load_signals(signals_path)
        self.reward_profile = reward_profile
        self.station_count = station_count
        self.battery_capacity = float(battery_capacity)
        self.max_power = float(max_power)
        self.slice_hours = float(slice_hours)
        self.initial_cost_norm = float(initial_cost_norm)
        self.initial_soc = float(initial_soc)
        self.reset()

    def reset(self) -> None:
        """重置到新的一天。"""
        self.t = 0
        self.charging_queue: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.metrics = EpisodeMetrics(powerlist=[[] for _ in range(self.station_count)])

    def build_initial_obs(self, duration_of_stay: int) -> np.ndarray:
        """为新到达车辆生成 11 维状态。"""
        pz = []
        for j in range(self.t + 1, self.t + 6):
            pz.append(float(self.signals.norm_price[j] if j < self.horizon else self.signals.norm_price[-1]))
        return np.asarray(
            [
                self.initial_soc,
                float(duration_of_stay) / 12.0,
                float(self.signals.pv[self.t]),
                float(self.signals.wt[self.t]),
                float(self.signals.load[self.t]),
                pz[0], pz[1], pz[2], pz[3], pz[4],
                self.initial_cost_norm,
            ],
            dtype=np.float32,
        )

    def enqueue(self, obs: Sequence[float], action: Sequence[float], station: int) -> None:
        """把一辆车加入当前时隙待更新队列。"""
        self.charging_queue.append((to_numpy_1d(obs), to_numpy_1d(action), int(station)))

    def _next_price_window(self) -> list[float]:
        """返回下一时刻能看到的 5 步前瞻价格窗口。"""
        out = []
        for j in range(self.t + 1, self.t + 6):
            out.append(float(self.signals.norm_price[j] if j < self.horizon else self.signals.norm_price[-1]))
        return out

    def _cost_upper_bound(self) -> float:
        """累计成本归一化上界，和原始实现保持一致。"""
        return float(self.signals.max_price * self.battery_capacity)

    def _dense_safety_penalty(self, soc: float, t_re: float, action: float) -> float:
        weight = float(self.reward_profile.dense_safety_penalty_weight)
        if weight <= 0.0:
            return 0.0
        step_delta = max(float(self.max_power * self.slice_hours / self.battery_capacity), 1e-8)
        remaining_slots = max(float(t_re) * 12.0, 1.0)
        target_soc = (
            float(self.reward_profile.exit_target_min)
            if self.reward_profile.dense_safety_target_soc is None
            else float(self.reward_profile.dense_safety_target_soc)
        )
        action_need = (target_soc - float(soc)) / (remaining_slots * step_delta)
        return float(max(0.0, action_need - float(action)) ** 2)

    def observation_bounds(self, *, max_duration_of_stay: int | float = 12.0) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative per-feature state bounds for adversarial projection."""
        duration_slots = max(float(max_duration_of_stay), 1.0)
        duration_norm_max = duration_slots / 12.0
        step_delta = float(self.max_power * self.slice_hours / self.battery_capacity)
        low = np.asarray(
            [
                float(self.initial_soc - duration_slots * step_delta),
                -1.0 / 12.0,
                float(np.min(self.signals.pv)),
                float(np.min(self.signals.wt)),
                float(np.min(self.signals.load)),
                float(np.min(self.signals.norm_price)),
                float(np.min(self.signals.norm_price)),
                float(np.min(self.signals.norm_price)),
                float(np.min(self.signals.norm_price)),
                float(np.min(self.signals.norm_price)),
                float(self.initial_cost_norm - duration_slots * step_delta),
            ],
            dtype=np.float32,
        )
        high = np.asarray(
            [
                float(self.initial_soc + duration_slots * step_delta),
                float(duration_norm_max),
                float(np.max(self.signals.pv)),
                float(np.max(self.signals.wt)),
                float(np.max(self.signals.load)),
                float(np.max(self.signals.norm_price)),
                float(np.max(self.signals.norm_price)),
                float(np.max(self.signals.norm_price)),
                float(np.max(self.signals.norm_price)),
                float(np.max(self.signals.norm_price)),
                float(self.initial_cost_norm + duration_slots * step_delta),
            ],
            dtype=np.float32,
        )
        return low, high

    def step(self) -> tuple[list[Transition], list[QueueItem], EpisodeMetrics]:
        """推进一个 15 分钟时隙。"""
        transitions: list[Transition] = []
        next_queue: list[QueueItem] = []
        total_power = 0.0
        power_by_station = [0.0 for _ in range(self.station_count)]

        for obs, action, station in self.charging_queue:
            obs = to_numpy_1d(obs)
            action = to_numpy_1d(action)
            a = float(action[0])
            soc = float(obs[0])
            t_re = float(obs[1])
            cost_norm = float(obs[10])

            # 关键物理更新：SOC 按动作线性变化，停留时间减一个时隙。
            new_soc = soc + a * self.max_power * self.slice_hours / self.battery_capacity
            new_t_re = t_re - 1.0 / 12.0
            next_idx = min(self.t + 1, self.horizon - 1)
            next_price = self._next_price_window()

            step_cost = a * self.max_power * self.slice_hours * float(self.signals.price[self.t])
            ncost = normalize_scalar(
                step_cost,
                -float(self.signals.max_price) * self.slice_hours * self.max_power * 0.5,
                float(self.signals.max_price) * self.slice_hours * self.max_power * 0.5,
            )
            cum_cost = float(min_max_denormalization(cost_norm, 0.0, self._cost_upper_bound())) + step_cost
            new_cost_norm = normalize_scalar(cum_cost, 0.0, self._cost_upper_bound())

            next_obs = np.asarray(
                [
                    new_soc,
                    new_t_re,
                    float(self.signals.pv[next_idx]),
                    float(self.signals.wt[next_idx]),
                    float(self.signals.load[next_idx]),
                    next_price[0], next_price[1], next_price[2], next_price[3], next_price[4],
                    new_cost_norm,
                ],
                dtype=np.float32,
            )

            done = bool(new_t_re < 1e-8)
            exit_penalty = 0.0
            running_penalty = 0.0
            exit_violation = 0
            running_violation = 0

            # 关键奖励逻辑：离站时要求接近目标 SOC，中途只要求安全区间。
            if done:
                if self.reward_profile.exit_target_min <= new_soc <= self.reward_profile.exit_target_max:
                    p_soc = 0.0
                elif new_soc > self.reward_profile.running_soc_max:
                    p_soc = 1.0 + new_soc - self.reward_profile.running_soc_max
                    exit_penalty = p_soc
                    exit_violation = 1
                    running_violation = 1
                else:
                    p_soc = 1.0 + self.reward_profile.exit_target_min - new_soc
                    exit_penalty = p_soc
                    exit_violation = 1
                    if new_soc < self.reward_profile.running_soc_min:
                        running_violation = 1
                self.metrics.costlist.append(float(cum_cost))
                self.metrics.final_soc_list.append(float(new_soc))
                self.metrics.done_count += 1
            else:
                if self.reward_profile.running_soc_min <= new_soc <= self.reward_profile.running_soc_max:
                    p_soc = 0.0
                elif new_soc > self.reward_profile.running_soc_max:
                    p_soc = 1.0 + new_soc - self.reward_profile.running_soc_max
                    running_penalty = p_soc
                    running_violation = 1
                else:
                    p_soc = 1.0 - new_soc + self.reward_profile.running_soc_min
                    running_penalty = p_soc
                    running_violation = 1
                next_queue.append(QueueItem(obs=next_obs.copy(), station=station))

            action_penalty = 0.0
            if self.reward_profile.action_penalty_threshold is not None and abs(a) > self.reward_profile.action_penalty_threshold:
                action_penalty = (abs(a) - self.reward_profile.action_penalty_threshold) * self.reward_profile.action_penalty_scale
            dense_safety_penalty = self._dense_safety_penalty(soc, t_re, a)
            reward = (
                -self.reward_profile.reward_soc_weight * p_soc
                - ncost
                - action_penalty
                - self.reward_profile.dense_safety_penalty_weight * dense_safety_penalty
            )
            total_power += a * self.max_power
            if 0 <= station < self.station_count:
                power_by_station[station] += a * self.max_power

            self.metrics.ep_reward += float(reward)
            self.metrics.ep_r1_cost_sum += float(step_cost)
            self.metrics.ep_r2_exit_penalty_sum += float(exit_penalty)
            self.metrics.ep_r3_running_penalty_sum += float(running_penalty)
            self.metrics.ep_r4_dense_safety_penalty_sum += float(dense_safety_penalty)
            self.metrics.exit_violation_count += int(exit_violation)
            self.metrics.running_violation_count += int(running_violation)
            self.metrics.total_transitions += 1

            transitions.append(
                Transition(
                    obs=obs.copy(),
                    action=np.asarray([a], dtype=np.float32),
                    reward=float(reward),
                    next_obs=next_obs.copy(),
                    done=done,
                    station=station,
                    step_cost=float(step_cost),
                    exit_penalty=float(exit_penalty),
                    running_penalty=float(running_penalty),
                    dense_safety_penalty=float(dense_safety_penalty),
                    exit_violation_count=int(exit_violation),
                    running_violation_count=int(running_violation),
                    final_soc=float(new_soc),
                )
            )

        self.charging_queue = []
        self.t += 1
        self.metrics.powercurve.append(float(total_power))
        for i in range(self.station_count):
            self.metrics.powerlist[i].append(float(power_by_station[i]))
        return transitions, next_queue, self.metrics


class Actor(nn.Module):
    """策略网络：11 维状态 -> 1 维连续动作。"""

    def __init__(self, obs_dim: int = 11, action_dim: int = 1, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.register_buffer('action_scale', torch.ones(action_dim, dtype=torch.float32))
        self.register_buffer('action_bias', torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播得到 [-1, 1] 动作。"""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class Critic(nn.Module):
    """Q 网络：输入状态和动作，输出 Q 值。"""

    def __init__(self, obs_dim: int = 11, action_dim: int = 1, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """计算 Q(s, a)。"""
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if act.ndim == 1:
            act = act.unsqueeze(0)
        x = torch.cat([obs.float(), act.float()], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """轻量级经验回放池，替代 stable_baselines3 依赖。"""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward: float, done: bool) -> None:
        """加入一条 transition。"""
        self.obs[self.pos] = to_numpy_1d(obs)
        self.next_obs[self.pos] = to_numpy_1d(next_obs)
        self.actions[self.pos] = to_numpy_1d(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """随机采样一个 batch。"""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'observations': torch.as_tensor(self.obs[idx], dtype=torch.float32, device=self.device),
            'next_observations': torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(self.actions[idx], dtype=torch.float32, device=self.device),
            'rewards': torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device).reshape(-1),
            'dones': torch.as_tensor(self.dones[idx], dtype=torch.float32, device=self.device).reshape(-1),
        }


class DDPGAgent:
    """基础 DDPG 代理。"""

    def __init__(
        self,
        actor: Actor,
        device: torch.device,
        gamma: float = 0.9,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
    ) -> None:
        self.device = device
        self.actor = actor.to(device)
        self.critic = Critic().to(device)
        self.actor_target = Actor().to(device)
        self.critic_target = Critic().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=float(critic_lr))
        self.gamma = float(gamma)
        self.tau = float(tau)

    @torch.no_grad()
    def act(self, obs, exploration_noise: float = 0.0, deterministic: bool = False) -> np.ndarray:
        """根据当前策略选择动作。"""
        obs_t = torch.as_tensor(to_numpy_1d(obs), dtype=torch.float32, device=self.device)
        action = self.actor(obs_t).reshape(-1)
        if not deterministic and exploration_noise > 0.0:
            noise = torch.normal(mean=0.0, std=float(exploration_noise), size=action.shape, device=self.device)
            action = action + noise
        return action.clamp(-1.0, 1.0).detach().cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor], *, freeze_actor: bool = False) -> dict[str, float]:
        """执行一次 DDPG 更新。"""
        with torch.no_grad():
            next_actions = self.actor_target(batch['next_observations'])
            q_target = batch['rewards'] + (1.0 - batch['dones']) * self.gamma * self.critic_target(
                batch['next_observations'], next_actions
            ).reshape(-1)

        q_pred = self.critic(batch['observations'], batch['actions']).reshape(-1)
        critic_loss = F.mse_loss(q_pred, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self.actor(batch['observations'])
        actor_loss = -self.critic(batch['observations'], actor_actions).mean()
        if not freeze_actor:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return {
            'critic_loss': float(critic_loss.detach().cpu().item()),
            'actor_loss': float(actor_loss.detach().cpu().item()),
            'mean_q': float(q_pred.detach().mean().cpu().item()),
        }

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        """软更新目标网络。"""
        for s, d in zip(src.parameters(), dst.parameters()):
            d.data.copy_(self.tau * s.data + (1.0 - self.tau) * d.data)



def save_actor(actor: Actor, path: str | Path) -> Path:
    """保存 actor 权重。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), path)
    return path


def save_baseline_bundle(agent: DDPGAgent, path: str | Path, *, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_type': 'baseline_bundle',
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'metadata': metadata or {},
        },
        path,
    )
    return path


def resolve_default_baseline_actor_path() -> Path:
    for candidate in (DEFAULT_BASELINE_ACTOR_PATH, LEGACY_BASELINE_ACTOR_PATH):
        if Path(candidate).exists():
            return Path(candidate)
    return Path(DEFAULT_BASELINE_ACTOR_PATH)


def _read_artifact_metadata_quietly(path: str | Path) -> dict:
    try:
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    except Exception:
        return {}
    if isinstance(payload, dict):
        return dict(payload.get('metadata') or {})
    return {}


def resolve_default_baseline_bundle_path(reward_profile: str | RewardProfile | None = None) -> Path:
    requested_profile = None
    if reward_profile is not None:
        requested_profile = (
            str(reward_profile.name)
            if isinstance(reward_profile, RewardProfile)
            else str(reward_profile)
        )
    baseline_dir = Path(DEFAULT_BASELINE_BUNDLE_PATH).parent
    discovered_candidates: list[Path] = []
    if baseline_dir.exists():
        discovered_candidates.extend(sorted(baseline_dir.glob('baseline_bundle*.pt')))
    ordered_candidates: list[Path] = []
    for candidate in (
        Path(DEFAULT_BASELINE_BUNDLE_PATH),
        Path(LEGACY_BASELINE_BUNDLE_PATH),
        *discovered_candidates,
    ):
        candidate = Path(candidate)
        if not candidate.exists() or candidate in ordered_candidates:
            continue
        ordered_candidates.append(candidate)
    if requested_profile is not None:
        for candidate in ordered_candidates:
            metadata = _read_artifact_metadata_quietly(candidate)
            if str(metadata.get('reward_profile')) == requested_profile:
                return candidate
    for candidate in ordered_candidates:
        return candidate
    return Path(DEFAULT_BASELINE_BUNDLE_PATH)


def _extract_actor_state_dict(payload) -> dict:
    """兼容原始 raw state_dict 与新项目的 actor payload。"""
    if isinstance(payload, dict) and 'state_dict' in payload and payload.get('model_type') == 'actor_state_dict':
        return payload['state_dict']
    if isinstance(payload, dict) and 'actor_state_dict' in payload:
        return payload['actor_state_dict']
    return payload


def load_actor_critic_bundle(path: str | Path, device: torch.device) -> dict:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    actor_state = _extract_actor_state_dict(payload)
    critic_state = None
    worst_critic_state = None
    adversary_state = None
    ppo_value_state = None
    ppo_policy_state = None
    adversary_value_state = None
    metadata = {}
    if isinstance(payload, dict):
        critic_state = payload.get('critic_state_dict')
        worst_critic_state = payload.get('worst_critic_state_dict')
        adversary_state = payload.get('adversary_state_dict')
        ppo_value_state = payload.get('ppo_value_state_dict')
        ppo_policy_state = payload.get('ppo_policy_state_dict')
        adversary_value_state = payload.get('adversary_value_state_dict')
        metadata = dict(payload.get('metadata') or {})
    return {
        'actor_state_dict': actor_state,
        'critic_state_dict': critic_state,
        'worst_critic_state_dict': worst_critic_state,
        'adversary_state_dict': adversary_state,
        'ppo_value_state_dict': ppo_value_state,
        'ppo_policy_state_dict': ppo_policy_state,
        'adversary_value_state_dict': adversary_value_state,
        'metadata': metadata,
    }


def load_actor_from_path(path: str | Path, device: torch.device) -> Actor:
    """从磁盘读取 actor 权重，兼容 legacy 和 merged 项目格式。"""
    actor = Actor().to(device)
    state = torch.load(Path(path), map_location=device, weights_only=False)
    actor.load_state_dict(_extract_actor_state_dict(state))
    actor.eval()
    return actor



def load_policy_actor(policy_mode: str, device: torch.device) -> Actor:
    """按关键字读取内置策略模型。当前仅保留 baseline。"""
    if policy_mode != 'baseline':
        raise ValueError(f'未知策略模式: {policy_mode}; 当前仅支持 baseline。')
    return load_actor_from_path(resolve_default_baseline_actor_path(), device)



def prepare_device(use_cuda: bool = True) -> torch.device:
    """优先返回 CUDA；不可用则回退到 CPU。"""
    return torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
