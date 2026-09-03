"""Minimal reader for one paired vehicle-and-signal scenario."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd


REQUIRED_VEHICLE_COLUMNS = ["Arrive_time", "Duration_of_stay", "Station"]


class PairedScenarioDataset:
    """Read the vehicle CSV and signal JSON belonging to one scenario."""

    def __init__(self, dataset_root: str | Path, split: str = "train", seed: int = 42, shuffle: bool = True):
        self.dataset_root = Path(dataset_root)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        manifest_path = self.dataset_root / "manifests" / "paired_scenario_manifest.csv"
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["Split"] == str(split)]
        if not self.rows:
            raise ValueError(f"No scenarios found for split={split!r}.")

    def __len__(self) -> int:
        return len(self.rows)

    def load_episode(self, episode_index: int) -> dict:
        cycle, position = divmod(int(episode_index), len(self.rows))
        rows = list(self.rows)
        if self.shuffle:
            random.Random(self.seed + cycle).shuffle(rows)
        row = rows[position]
        vehicle_path = self.dataset_root / row["Vehicle_File"]
        arrivals = pd.read_csv(vehicle_path)
        missing = set(REQUIRED_VEHICLE_COLUMNS).difference(arrivals.columns)
        if missing:
            raise ValueError(f"{vehicle_path} is missing columns: {sorted(missing)}")
        return {
            "scenario_id": row["Scenario_ID"],
            "arrivals": arrivals[REQUIRED_VEHICLE_COLUMNS].astype(int).sort_values(["Arrive_time", "Station"]).reset_index(drop=True),
            "signal_path": self.dataset_root / row["Signal_File"],
            "metadata": row,
        }
