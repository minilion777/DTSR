"""Run the leak-free K0--K4 knowledge-ladder protocol.

The legacy Experiment-4 evaluator is loaded without modification so that its
dataset pairing and metric definitions remain identical.  This wrapper replaces
only the attacker factory and enforces a single, precommitted restart.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.causal_knowledge_ladder_attacks import CausalKnowledgeConfig, CausalKnowledgeLadderAttacker
from evc.long_horizon_attacks import build_long_horizon_attacker
from evc.merged_attacks import build_state_attacker


def _load_legacy_evaluator():
    target = SCRIPT_DIR / "21_evaluate_exp4_adaptive_long_horizon_seed42.py"
    spec = importlib.util.spec_from_file_location("_legacy_exp4_evaluator", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluator: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    evaluator = _load_legacy_evaluator()
    captured: dict[str, object] = {}
    legacy_parse_args = evaluator.parse_args

    def parse_args_with_causal_guard():
        args = legacy_parse_args()
        if int(args.restarts) != 1:
            raise ValueError(
                "Causal knowledge-ladder protocol requires --restarts 1. "
                "It never selects an attack using post-rollout test outcomes."
            )
        captured["args"] = args
        return args

    def build_causal_attacker(*, condition, actor, critic, device, arrivals, signal_path, attack_seed, dae,
                               detector_model, detector_threshold, shield_config, ug_bcr_config,
                               ug_bcr_v3_config, args):
        del critic, arrivals, signal_path  # Explicitly reject test-day simulator/signal access.
        if ug_bcr_v3_config is not None:
            raise ValueError("Causal knowledge-ladder protocol supports audited UG-BCR-v2 only.")
        # The PGD anchor is actor-only.  All module knowledge is introduced by
        # CausalKnowledgeLadderAttacker according to condition["knowledge"].
        base = build_state_attacker(
            actor,
            device=device,
            algorithm="opposite_pgd",
            epsilon=min(0.025, float(args.epsilon)),
            alpha=0.006,
            iters=1,
            seed=int(attack_seed),
            obs_low=None,
            obs_high=None,
            attack_state_scope=str(condition["scope"]),
        )
        # K0 is the experimental control: it is the ordinary non-adaptive
        # Deadline-PGD used in the strength study, with no defense component
        # passed to it.  Its standard temporal dynamics are public attack
        # behavior, not a shadow copy of the target defense.
        if str(condition["knowledge"]).upper() == "K0":
            effective_epsilon = min(0.025, float(args.epsilon))
            scale = effective_epsilon / 0.055
            return build_long_horizon_attacker(
                "local_deadline_drift_pgd",
                actor=actor,
                device=device,
                # All evaluated observation coordinates are normalized.  The
                # legacy evaluator's state attacker also uses this [0, 1]
                # projection when no scenario-specific bounds are supplied.
                obs_low=None,
                obs_high=None,
                critic=None,
                seed=int(attack_seed),
                attack_state_scope="local",
                attack_overrides={
                    "epsilon": effective_epsilon,
                    "base_epsilon": 0.028 * scale,
                    "base_alpha": 0.008 * scale,
                    "base_iters": 5,
                },
            )
        attacker = CausalKnowledgeLadderAttacker(
            base,
            attack_state_scope=str(condition["scope"]),
            config=CausalKnowledgeConfig(
                knowledge_level=str(condition["knowledge"]),
                # Keep the actual Linf budget identical at every knowledge
                # level.  Knowledge, rather than a larger perturbation, is the
                # only permitted source of stronger K1--K4 attacks.
                epsilon=min(0.025, float(args.epsilon)),
                temporal_eta=float(args.temporal_eta),
                # Fixed public protocol constant, never derived from test signals.
                public_cost_upper_bound=1.0,
            ),
        )
        attacker.configure_target_defense(
            defender=dae,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            shield_config=shield_config,
            ug_bcr_config=ug_bcr_config,
            reward_profile=evaluator.TRAIN_PROFILE,
            device=device,
            actor=actor,
            repair_mode=evaluator.REPAIR_MODE,
        )
        return attacker

    def select_only_precommitted_restart(raw_df):
        # All jobs use restart zero.  Keeping this explicit prevents a future
        # refactor from ranking candidates by full-DTSR test reward/violations.
        return raw_df[raw_df["condition_key"] != "clean"].copy()

    evaluator.parse_args = parse_args_with_causal_guard
    evaluator.build_exp4_attacker = build_causal_attacker
    evaluator.select_restart_per_condition = select_only_precommitted_restart
    evaluator.main()

    args = captured.get("args")
    if args is None:
        raise RuntimeError("Causal evaluator did not capture parsed arguments.")
    output_dir = Path(args.output_dir)
    protocol = {
        "attacker": "causal_knowledge_ladder",
        "knowledge_contract": {
            "K0": "Actor only; ordinary local Deadline-PGD control at epsilon=0.025",
            "K1": "Actor + DAE; deterministic always-DAE shadow route, no oracle label",
            "K2": "K1 + DET and its threshold",
            "K3": "K2 + UG-BCR; Shield configuration is withheld and passed as None",
            "K4": "K3 + Temporal Shield",
        },
        "causal_guarantees": [
            "No signals_path is accepted by the attacker API.",
            "No future PV, WT, load, or price values are read.",
            "The only price retained is the current/past raw_price in AttackContext.",
            "Restart count is fixed to one; no post-rollout test-result selection is performed.",
        ],
        "public_cost_upper_bound": 1.0,
        "effective_linf_budget": 0.025,
        "restarts": 1,
    }
    (output_dir / "causal_knowledge_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # The reused evaluator labels its historical attacker in run_config.json.
    # Correct the produced artifact so downstream plotting/reporting cannot
    # accidentally describe these measurements as the legacy CEM-MPC attack.
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config["attacker"] = "causal_knowledge_ladder"
        run_config["restarts"] = 1
        run_config["restart_selection"] = "fixed restart=0; no post-rollout outcome selection"
        run_config["causal_protocol_manifest"] = "causal_knowledge_protocol.json"
        run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Causal protocol manifest: {output_dir / 'causal_knowledge_protocol.json'}")


if __name__ == "__main__":
    main()
