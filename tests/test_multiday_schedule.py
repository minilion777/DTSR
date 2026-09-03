from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evc.multiday_schedule import (
    EpisodeScenario,
    max_duration_across_scenarios,
    normalize_episode_scenarios,
    scenario_for_episode,
)


def arrivals(duration: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Arrive_time": [1],
            "Duration_of_stay": [duration],
            "Station": [0],
        }
    )


class MultidayScheduleTests(unittest.TestCase):
    def test_single_scenario_fallback_is_backward_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            signal_path = Path(directory) / "signals.json"
            signal_path.write_text("{}", encoding="utf-8")
            schedule = normalize_episode_scenarios(arrivals(4), signal_path, None)
            self.assertEqual(len(schedule), 1)
            self.assertEqual(schedule[0].scenario_id, "fixed_scenario")

    def test_round_robin_schedule_and_global_duration_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            signal_path = Path(directory) / "signals.json"
            signal_path.write_text("{}", encoding="utf-8")
            schedule = normalize_episode_scenarios(
                arrivals(4),
                signal_path,
                [
                    EpisodeScenario(arrivals(6), signal_path, "train_a"),
                    EpisodeScenario(arrivals(12), signal_path, "train_b"),
                ],
            )
            self.assertEqual(scenario_for_episode(schedule, 1).scenario_id, "train_a")
            self.assertEqual(scenario_for_episode(schedule, 2).scenario_id, "train_b")
            self.assertEqual(scenario_for_episode(schedule, 3).scenario_id, "train_a")
            self.assertEqual(max_duration_across_scenarios(schedule), 12.0)

    def test_empty_schedule_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            signal_path = Path(directory) / "signals.json"
            signal_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                normalize_episode_scenarios(arrivals(4), signal_path, [])


if __name__ == "__main__":
    unittest.main()
