from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import (
    DEFAULT_ACTOR_PATH,
    DEFAULT_BUNDLE_PATH,
    actor_matches_bundle,
    PACKAGE_ROOT,
    deterministic_subset,
    load_manifest,
    load_scenario,
    resolve_device,
)

sys.path.insert(0, str(PACKAGE_ROOT))
from dtsr_multiday_common import REPAIR_MODE
from evc.defense import load_dae, load_detector
from evc.merged_attacks import PGDStateAttacker
from evc.merged_core import ChargingEnv, Critic, TRAIN_PROFILE, load_actor_critic_bundle, load_actor_from_path
from evc.offline_dae_det_temporal_shield import load_temporal_shield_bundle
from evc.ug_bcr import BeliefCoreConfig, UGBCRConfig, UrgencyGateConfig, rollout_episode_with_ug_bcr


def scalar_summary(summary):
    return {key: value for key, value in summary.items() if not isinstance(value, list)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--actor-path", type=Path, default=DEFAULT_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--algorithm", default="opposite_pgd", choices=["opposite_pgd", "q_function"])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--state-scope", choices=["local", "all"], default="all")
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "results" / "dtsr_evaluation.csv")
    args = parser.parse_args()

    device = resolve_device(args.device)
    actor = load_actor_from_path(args.actor_path, device)
    payload = load_actor_critic_bundle(args.bundle_path, device)
    if payload.get("critic_state_dict") is None:
        raise RuntimeError("Selected bundle does not contain critic weights.")
    if not actor_matches_bundle(actor, payload):
        raise RuntimeError(
            "--actor-path and --bundle-path do not contain the same actor weights."
        )
    critic = Critic().to(device)
    critic.load_state_dict(payload["critic_state_dict"])
    critic.eval()

    dae = load_dae(args.dtsr_dir / "dtsr_dae.pt", device)
    detector_artifact = load_detector(args.dtsr_dir / "dtsr_detector.pt", device)
    shield_artifact = load_temporal_shield_bundle(args.dtsr_dir / "dtsr_temporal_shield.pt")
    ug_config = UGBCRConfig(
        belief=BeliefCoreConfig(enabled=True),
        urgency_gate=UrgencyGateConfig(enabled=True),
    )

    manifest = deterministic_subset(load_manifest(args.split), args.scenes, 42)
    rows = []
    for _, row in manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        env = ChargingEnv(signal_path, TRAIN_PROFILE)
        low, high = env.observation_bounds(max_duration_of_stay=12)
        attacker = PGDStateAttacker(
            actor,
            device=device,
            algorithm=args.algorithm,
            epsilon=args.epsilon,
            alpha=0.01,
            iters=10,
            obs_low=low,
            obs_high=high,
            critic=critic if args.algorithm == "q_function" else None,
            attack_state_scope=args.state_scope,
        )

        clean = rollout_episode_with_ug_bcr(
            arrivals, actor, signal_path, device, TRAIN_PROFILE,
            attack_enabled=False,
            route_mode="none",
            label="clean",
        )
        attacked = rollout_episode_with_ug_bcr(
            arrivals, actor, signal_path, device, TRAIN_PROFILE,
            attack_enabled=True,
            attack_scenario="O",
            attacker=attacker.clone(),
            route_mode="none",
            state_scope=args.state_scope,
            attack_scope="obs",
            label="attack",
        )
        defended = rollout_episode_with_ug_bcr(
            arrivals, actor, signal_path, device, TRAIN_PROFILE,
            attack_enabled=True,
            attack_scenario="O",
            attacker=attacker.clone(),
            defender=dae,
            detector_model=detector_artifact.model,
            detector_threshold=detector_artifact.threshold,
            shield_config=shield_artifact.config,
            route_mode="detector",
            enable_shield=True,
            enable_belief=True,
            enable_urgency_gate=True,
            ug_bcr_config=ug_config,
            state_scope=args.state_scope,
            attack_scope="obs",
            label="dtsr",
            repair_mode=REPAIR_MODE,
        )
        for label, summary in [("clean", clean), ("attack", attacked), ("dtsr", defended)]:
            item = scalar_summary(summary)
            item.update({"scenario_id": scenario_id, "condition": label, "algorithm": args.algorithm, "epsilon": args.epsilon})
            rows.append(item)
        print(f"Completed {scenario_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
