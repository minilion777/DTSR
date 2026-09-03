from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

DATASET_ROOT = PACKAGE_ROOT / "multiday_dataset"
MANIFEST_PATH = DATASET_ROOT / "manifests" / "paired_scenario_manifest.csv"
DEFAULT_ACTOR_PATH = PACKAGE_ROOT / "models" / "baseline" / "actor_baseline_ep50_seed42.pt"
DEFAULT_BUNDLE_PATH = PACKAGE_ROOT / "models" / "baseline" / "baseline_bundle_ep50_seed42.pt"
REFERENCE_DATA_PATH = PACKAGE_ROOT / "data" / "data_reference.csv"
REFERENCE_SIGNAL_PATH = PACKAGE_ROOT / "data" / "signals_reference.json"


def resolve_device(device_text: str) -> torch.device:
    token = str(device_text).lower().strip()
    if token == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if token == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(token)


def load_manifest(split: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST_PATH)
    if split is not None:
        frame = frame[frame["Split"] == split].copy()
    return frame.reset_index(drop=True)


def deterministic_subset(frame: pd.DataFrame, count: int | None, seed: int) -> pd.DataFrame:
    if count is None or int(count) <= 0 or int(count) >= len(frame):
        return frame.copy().reset_index(drop=True)
    order = list(range(len(frame)))
    random.Random(int(seed)).shuffle(order)
    return frame.iloc[order[: int(count)]].reset_index(drop=True)


def load_scenario(row) -> tuple[pd.DataFrame, Path, str]:
    arrivals = pd.read_csv(DATASET_ROOT / row["Vehicle_File"])
    arrivals = arrivals[["Arrive_time", "Duration_of_stay", "Station"]].astype(int)
    arrivals = arrivals.sort_values(["Arrive_time", "Station"]).reset_index(drop=True)
    return arrivals, DATASET_ROOT / row["Signal_File"], str(row["Scenario_ID"])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def actor_matches_bundle(actor: torch.nn.Module, bundle_payload: dict) -> bool:
    bundle_state = bundle_payload.get("actor_state_dict")
    if bundle_state is None:
        return False
    actor_state = actor.state_dict()
    if set(actor_state) != set(bundle_state):
        return False
    return all(
        torch.equal(
            actor_state[key].detach().cpu(),
            bundle_state[key].detach().cpu(),
        )
        for key in actor_state
    )
