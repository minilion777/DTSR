from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_VEHICLE_COLUMNS = [
    "Arrive_time",
    "Duration_of_stay",
    "Station",
]


class PairedScenarioDataset:
    """Load vehicle, signal and 96-slot context files as one scenario."""

    def __init__(
        self,
        dataset_root,
        split="train",
        seed=42,
        shuffle=True,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

        manifest_path = (
            self.dataset_root
            / "manifests"
            / "paired_scenario_manifest.csv"
        )
        with manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            rows = list(csv.DictReader(f))

        self.rows = [
            row for row in rows
            if row["Split"] == self.split
        ]
        if not self.rows:
            raise ValueError(
                f"No scenarios found for split={self.split}"
            )

    def __len__(self):
        return len(self.rows)

    def _row_for_episode(self, episode_index):
        episode_index = int(episode_index)
        cycle = episode_index // len(self.rows)
        position = episode_index % len(self.rows)

        rows = list(self.rows)
        if self.shuffle:
            random.Random(self.seed + cycle).shuffle(rows)
        return rows[position]

    def load_episode(self, episode_index):
        row = self._row_for_episode(episode_index)

        vehicle_path = self.dataset_root / row["Vehicle_File"]
        signal_path = self.dataset_root / row["Signal_File"]
        context_path = self.dataset_root / row["Context_File"]

        arrivals = pd.read_csv(vehicle_path)
        missing = [
            col for col in REQUIRED_VEHICLE_COLUMNS
            if col not in arrivals.columns
        ]
        if missing:
            raise ValueError(
                f"{vehicle_path} is missing columns: {missing}"
            )

        arrivals = arrivals[
            REQUIRED_VEHICLE_COLUMNS
        ].astype(int)
        arrivals = arrivals.sort_values(
            ["Arrive_time", "Station"]
        ).reset_index(drop=True)

        context = pd.read_csv(context_path)
        if len(context) != 96:
            raise ValueError(
                f"{context_path} must have 96 time slots"
            )

        return {
            "scenario_id": row["Scenario_ID"],
            "arrivals": arrivals,
            "signal_path": signal_path,
            "context": context,
            "metadata": row,
        }


def build_initial_observation(
    *,
    duration_of_stay,
    context,
    time_slot,
    initial_soc=0.0,
    initial_cost_norm=0.2,
):
    """Reproduce ChargingEnv.build_initial_obs exactly."""
    row = context.iloc[int(time_slot)]
    return np.asarray(
        [
            float(initial_soc),
            float(duration_of_stay) / 12.0,
            float(row["PV_norm"]),
            float(row["WT_norm"]),
            float(row["Load_norm"]),
            float(row["Price_t_plus_1_norm"]),
            float(row["Price_t_plus_2_norm"]),
            float(row["Price_t_plus_3_norm"]),
            float(row["Price_t_plus_4_norm"]),
            float(row["Price_t_plus_5_norm"]),
            float(initial_cost_norm),
        ],
        dtype=np.float32,
    )
