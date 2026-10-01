"""Evaluate DTSR against the paper short- and long-horizon attacks."""

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

from evc.attacks import (
    ALL_PAPER_ATTACK_NAMES,
    LONG_HORIZON_ATTACK_NAMES,
    SHORT_HORIZON_ATTACK_SPECS,
    build_long_horizon_attacker,
    build_short_horizon_attacker,
)
from evc.defense import load_dae, load_detector
from evc.dtsr_runtime import rollout_episode_with_dtsr
from evc.merged_core import ChargingEnv, Critic, TRAIN_PROFILE, load_actor_critic_bundle, load_actor_from_path
from evc.offline_dae_det_temporal_shield import load_temporal_shield_bundle
from evc.pd_bcr import PDBCRConfig, load_pd_bcr_config, rollout_episode_with_pd_bcr


REPAIR_MODE = "full"


def scalar_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not isinstance(value, list)}


def parse_attacks(raw: str) -> list[str]:
    attacks = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(attacks) - set(ALL_PAPER_ATTACK_NAMES))
    if unknown:
        raise ValueError(f"Unsupported attacks: {unknown}")
    if not attacks:
        raise ValueError("At least one attack is required.")
    return list(dict.fromkeys(attacks))


def build_attack(name, *, actor, critic, device, low, high, seed):
    if name in SHORT_HORIZON_ATTACK_SPECS:
        spec = SHORT_HORIZON_ATTACK_SPECS[name]
        return (
            build_short_horizon_attacker(
                name,
                actor=actor,
                critic=critic,
                device=device,
                obs_low=low,
                obs_high=high,
                seed=seed,
            ),
            spec.scenario,
            spec.state_scope,
            "short_horizon",
        )
    if name in LONG_HORIZON_ATTACK_NAMES:
        return (
            build_long_horizon_attacker(
                name,
                actor=actor,
                critic=critic,
                device=device,
                obs_low=low,
                obs_high=high,
                seed=seed,
                attack_state_scope="local",
            ),
            "O",
            "local",
            "long_horizon",
        )
    raise ValueError(f"Unsupported attack: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DAE + DeT + Temporal Shield + PD-BCR against all retained paper attacks."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--actor-path", type=Path, default=DEFAULT_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "runs" / "dtsr")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attacks", default=",".join(ALL_PAPER_ATTACK_NAMES))
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "runs" / "evaluation" / "dtsr_attacks.csv",
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
    pd_bcr_path = args.dtsr_dir / "pd_bcr_config.json"
    pd_bcr_config = load_pd_bcr_config(pd_bcr_path) if pd_bcr_path.exists() else PDBCRConfig()

    rows: list[dict] = []
    manifest = deterministic_subset(load_manifest(args.split), args.scenes, args.seed)
    for scene_index, (_, row) in enumerate(manifest.iterrows()):
        arrivals, signal_path, scenario_id = load_scenario(row)
        env = ChargingEnv(signal_path, TRAIN_PROFILE)
        low, high = env.observation_bounds(max_duration_of_stay=12)
        clean = rollout_episode_with_dtsr(
            arrivals,
            actor,
            signal_path,
            device,
            TRAIN_PROFILE,
            attack_enabled=False,
            route_mode="none",
            label="clean",
        )

        for attack_index, attack_name in enumerate(attacks):
            attack_seed = int(args.seed + scene_index * 10_000 + attack_index * 1_000)
            attacker, attack_scenario, state_scope, attack_type = build_attack(
                attack_name,
                actor=actor,
                critic=critic,
                device=device,
                low=low,
                high=high,
                seed=attack_seed,
            )
            attacked = rollout_episode_with_dtsr(
                arrivals,
                actor,
                signal_path,
                device,
                TRAIN_PROFILE,
                attack_enabled=True,
                attack_scenario=attack_scenario,
                attacker=attacker.clone(),
                route_mode="none",
                state_scope=state_scope,
                attack_scope="obs",
                label="attack",
            )
            defended = rollout_episode_with_pd_bcr(
                arrivals,
                actor,
                signal_path,
                device,
                TRAIN_PROFILE,
                attack_enabled=True,
                attack_scenario=attack_scenario,
                attacker=attacker.clone(),
                defender=dae,
                detector_model=detector_artifact.model,
                detector_threshold=detector_artifact.threshold,
                shield_config=shield_artifact.config,
                route_mode="detector",
                pd_bcr_config=pd_bcr_config,
                state_scope=state_scope,
                attack_scope="obs",
                label="dtsr",
                repair_mode=REPAIR_MODE,
            )
            for condition, summary in (("clean", clean), ("attack", attacked), ("dtsr", defended)):
                item = scalar_summary(summary)
                item.update(
                    scenario_id=scenario_id,
                    condition=condition,
                    attack=attack_name,
                    attack_scenario=attack_scenario,
                    attack_seed=attack_seed,
                    attack_type=attack_type,
                    state_scope=state_scope,
                )
                rows.append(item)
        print(f"Completed {scenario_id}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
