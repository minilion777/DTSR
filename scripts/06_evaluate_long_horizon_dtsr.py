"""Evaluate the complete DTSR defense against the retained long-horizon attacks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from _common import (
    DEFAULT_ACTOR_PATH,
    DEFAULT_BUNDLE_PATH,
    PACKAGE_ROOT,
    actor_matches_bundle,
    deterministic_subset,
    load_manifest,
    load_scenario,
    resolve_device,
)

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import load_dae, load_detector
from evc.long_horizon_attacks import build_long_horizon_attacker
from evc.merged_core import ChargingEnv, Critic, TRAIN_PROFILE, load_actor_critic_bundle, load_actor_from_path
from evc.offline_dae_det_temporal_shield import load_temporal_shield_bundle
from evc.ug_bcr import BeliefCoreConfig, UGBCRConfig, UrgencyGateConfig, rollout_episode_with_ug_bcr


# These are the two non-adaptive, stateful long-horizon attacks retained in
# this release.  Adaptive/CEM-MPC variants belong to removed extension studies.
SUPPORTED_ATTACKS = ("local_small_drift_q", "local_deadline_drift_pgd")
REPAIR_MODE = "full"


def scalar_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not isinstance(value, list)}


def parse_attacks(raw: str) -> list[str]:
    attacks = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(attacks) - set(SUPPORTED_ATTACKS))
    if unknown:
        raise ValueError(f"Unsupported retained long-horizon attacks: {unknown}")
    if not attacks:
        raise ValueError("At least one long-horizon attack is required.")
    return list(dict.fromkeys(attacks))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate full DAE + DeT + Temporal Shield + UG-BCR DTSR against long-horizon attacks."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--actor-path", type=Path, default=DEFAULT_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "runs" / "dtsr")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attacks", default=",".join(SUPPORTED_ATTACKS))
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "runs" / "evaluation" / "dtsr_long_horizon.csv",
    )
    args = parser.parse_args()
    attacks = parse_attacks(args.attacks)

    device = resolve_device(args.device)
    actor = load_actor_from_path(args.actor_path, device).eval()
    payload = load_actor_critic_bundle(args.bundle_path, device)
    if payload.get("critic_state_dict") is None or not actor_matches_bundle(actor, payload):
        raise RuntimeError("--actor-path and --bundle-path must be matching DDPG checkpoints with critic weights.")
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

    rows: list[dict] = []
    manifest = deterministic_subset(load_manifest(args.split), args.scenes, args.seed)
    for scene_index, (_, row) in enumerate(manifest.iterrows()):
        arrivals, signal_path, scenario_id = load_scenario(row)
        env = ChargingEnv(signal_path, TRAIN_PROFILE)
        low, high = env.observation_bounds(max_duration_of_stay=12)
        clean = rollout_episode_with_ug_bcr(
            arrivals, actor, signal_path, device, TRAIN_PROFILE,
            attack_enabled=False, route_mode="none", label="clean",
        )
        for attack_index, attack_name in enumerate(attacks):
            attack_seed = int(args.seed + scene_index * 10_000 + attack_index * 1_000)
            attacker = build_long_horizon_attacker(
                attack_name,
                actor=actor,
                critic=critic,
                device=device,
                obs_low=low,
                obs_high=high,
                seed=attack_seed,
                attack_state_scope="local",
            )
            attacked = rollout_episode_with_ug_bcr(
                arrivals, actor, signal_path, device, TRAIN_PROFILE,
                attack_enabled=True, attack_scenario="O", attacker=attacker.clone(),
                route_mode="none", state_scope="local", attack_scope="obs", label="attack",
            )
            defended = rollout_episode_with_ug_bcr(
                arrivals, actor, signal_path, device, TRAIN_PROFILE,
                attack_enabled=True, attack_scenario="O", attacker=attacker.clone(),
                defender=dae, detector_model=detector_artifact.model,
                detector_threshold=detector_artifact.threshold,
                shield_config=shield_artifact.config, route_mode="detector",
                enable_shield=True, enable_belief=True, enable_urgency_gate=True,
                ug_bcr_config=ug_config, state_scope="local", attack_scope="obs",
                label="dtsr", repair_mode=REPAIR_MODE,
            )
            for condition, summary in (("clean", clean), ("attack", attacked), ("dtsr", defended)):
                item = scalar_summary(summary)
                item.update(
                    scenario_id=scenario_id,
                    condition=condition,
                    algorithm=attack_name,
                    attack_seed=attack_seed,
                    attack_type="long_horizon",
                )
                rows.append(item)
        print(f"Completed {scenario_id}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
