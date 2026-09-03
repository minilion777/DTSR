from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeScenario:
    arrivals: pd.DataFrame
    signals_path: Path
    scenario_id: str


def normalize_episode_scenarios(
    arrivals: pd.DataFrame,
    signals_path: str | Path,
    episode_scenarios: Sequence[EpisodeScenario | tuple] | None,
) -> tuple[EpisodeScenario, ...]:
    if episode_scenarios is None:
        return (EpisodeScenario(arrivals, Path(signals_path), 'fixed_scenario'),)

    normalized: list[EpisodeScenario] = []
    for index, item in enumerate(episode_scenarios):
        if isinstance(item, EpisodeScenario):
            scenario = item
        else:
            if len(item) != 3:
                raise ValueError('Each episode scenario must contain arrivals, signals_path, and scenario_id.')
            scenario = EpisodeScenario(item[0], Path(item[1]), str(item[2]))
        required = {'Arrive_time', 'Duration_of_stay', 'Station'}
        missing = required.difference(scenario.arrivals.columns)
        if missing:
            raise ValueError(f'Episode scenario {index} is missing arrival columns: {sorted(missing)}')
        if not scenario.signals_path.exists():
            raise FileNotFoundError(f'Episode scenario signal file not found: {scenario.signals_path}')
        normalized.append(scenario)
    if not normalized:
        raise ValueError('episode_scenarios must not be empty.')
    return tuple(normalized)


def scenario_for_episode(scenarios: Sequence[EpisodeScenario], episode: int) -> EpisodeScenario:
    if episode <= 0:
        raise ValueError('episode must be positive.')
    return scenarios[(int(episode) - 1) % len(scenarios)]


def max_duration_across_scenarios(scenarios: Sequence[EpisodeScenario], fallback: float = 12.0) -> float:
    values = [
        float(np.max(np.asarray(item.arrivals['Duration_of_stay'], dtype=np.float32)))
        for item in scenarios
        if len(item.arrivals) > 0
    ]
    return max(values, default=float(fallback))
