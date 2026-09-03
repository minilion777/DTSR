from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import DATASET_ROOT, MANIFEST_PATH, PACKAGE_ROOT, actor_matches_bundle, load_manifest, load_scenario, resolve_device, write_json
from dtsr_multiday_common import union_observation_bounds

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from cli import posterior_detector_dataset_from_unified_pair
from evc.defense import DenoisingAutoencoder
from evc.long_horizon_attacks import FullPipelineAdaptiveDeadlineAttacker
from evc.merged_pipeline import PairDatasetBundle
from evc.offline_dae_det_temporal_shield import LOCAL_SHIELD_INDICES
from evc.merged_core import load_actor_critic_bundle, load_actor_from_path
from evc.ug_bcr import rollout_episode_with_ug_bcr

DEFAULT_ACTOR = PACKAGE_ROOT / "models" / "multiday_ddpg_baseline_bundle" / "actor_multiday_best.pt"
DEFAULT_BUNDLE = PACKAGE_ROOT / "models" / "multiday_ddpg_baseline_bundle" / "bundle_multiday_best.pt"
REFERENCE_HASH_PATH = PACKAGE_ROOT / "config" / "original_logic_reference.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_positions(source: str, tokens: list[str]) -> dict[str, int]:
    positions: dict[str, int] = {}
    cursor = 0
    for token in tokens:
        pos = source.find(token, cursor)
        if pos < 0:
            raise AssertionError(f"Pipeline token not found after position {cursor}: {token}")
        positions[token] = pos
        cursor = pos + len(token)
    return positions


def audit_original_logic_hashes() -> dict:
    expected = json.loads(REFERENCE_HASH_PATH.read_text(encoding="utf-8"))
    rows = []
    for relative, expected_hash in expected.items():
        path = PACKAGE_ROOT / relative
        actual = sha256(path)
        rows.append(
            {
                "file": relative,
                "expected_sha256": expected_hash,
                "actual_sha256": actual,
                "matches_original_reference": actual == expected_hash,
            }
        )
    mismatches = [row for row in rows if not row["matches_original_reference"]]
    if mismatches:
        raise AssertionError(f"Core implementation drifted from original reference: {mismatches}")
    return {"files": rows, "all_match": True}


def audit_pipeline_order() -> dict:
    runtime_source = inspect.getsource(rollout_episode_with_ug_bcr)
    adaptive_source = inspect.getsource(FullPipelineAdaptiveDeadlineAttacker._target_pipeline_eval)

    # Original implementation contract.  The additive ablation names
    # DAE -> DAE+DET -> +Shield -> +UG-BCR do NOT imply call order inside the
    # full pipeline.  The original code selects the policy/belief branch first,
    # then applies Temporal Shield to the selected state.
    runtime_tokens = [
        "route_fn(",
        "belief.repair_batch(",
        "urgency_gate.select_batch(",
        "_apply_shield_batch(",
        "_compute_actions(",
    ]
    adaptive_tokens = [
        "route_fn(",
        "belief.repair_batch(",
        "gate.select_batch(",
        "_shield_single_state(",
        "_actor_action_on_state(",
    ]

    runtime_positions = _ordered_positions(runtime_source, runtime_tokens)
    adaptive_positions = _ordered_positions(adaptive_source, adaptive_tokens)

    return {
        "runtime_contract": "DAE/DET route -> UG-BCR belief+urgency gate -> Temporal Shield -> Actor",
        "adaptive_shadow_contract": "DAE/DET route -> UG-BCR belief+urgency gate -> Temporal Shield -> Actor",
        "ablation_addition_order": "Attack -> Denoise -> Denoise+DET -> +Shield -> +UG-BCR",
        "runtime_positions": runtime_positions,
        "adaptive_positions": adaptive_positions,
        "runtime_shadow_match": True,
    }


def audit_dataset() -> dict:
    expected = {"train": 500, "val": 60, "test": 120}
    result: dict[str, dict] = {}
    full = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig")
    assert full["Scenario_ID"].is_unique, "Scenario_ID must be unique in paired manifest."

    split_sets: dict[str, set[str]] = {}
    for split, expected_count in expected.items():
        frame = load_manifest(split)
        assert len(frame) == expected_count, (split, len(frame), expected_count)
        split_sets[split] = set(frame["Scenario_ID"].astype(str))
        missing = []
        for row in frame.to_dict(orient="records"):
            for key in ("Vehicle_File", "Signal_File", "Context_File"):
                path = DATASET_ROOT / str(row[key])
                if not path.exists():
                    missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"Missing {split} scenario files: {missing[:10]}")
        result[split] = {"scenario_count": len(frame), "missing_file_count": 0}

    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    result["split_isolation"] = {"train_val_test_disjoint": True}
    return result



def audit_posterior_runtime_alignment(actor_path: Path, device: torch.device) -> dict:
    actor = load_actor_from_path(actor_path, device).eval()
    rng = np.random.default_rng(42)
    clean = rng.uniform(0.05, 0.95, size=(24, 11)).astype(np.float32)
    adv = clean.copy()
    adv[:, 0] = np.clip(adv[:, 0] + 0.08, 0.0, 1.2)
    adv[:, 1] = np.clip(adv[:, 1] - 0.05, 0.0, 1.2)
    adv[:, 10] = np.clip(adv[:, 10] + 0.04, 0.0, 1.2)
    # Also perturb a global feature. Full-state DAE routing must evaluate the
    # same 11-dimensional reconstructed candidate that runtime sends onward.
    adv[:, 5] = np.clip(adv[:, 5] + 0.07, 0.0, 1.2)
    episode_indices = np.repeat(np.arange(3, dtype=np.int64), 8)
    vehicle_ids = np.tile(np.arange(8, dtype=np.int64), 3)
    bundle = PairDatasetBundle(
        adv_inputs=adv,
        clean_inputs=clean,
        metadata={"audit": True},
        time_indices=np.tile(np.arange(8, dtype=np.int64), 3),
        stations=np.zeros((24,), dtype=np.int64),
        is_new_arrivals=np.zeros((24,), dtype=np.int64),
        vehicle_ids=vehicle_ids,
        episode_indices=episode_indices,
        attack_mask=np.ones((24,), dtype=np.int64),
    )
    dae = DenoisingAutoencoder(input_dim=11, seq_len=8).to(device).eval()
    detector_bundle = posterior_detector_dataset_from_unified_pair(
        bundle,
        actor,
        dae,
        device,
        profile_tag="integrity_audit",
        train_attack_tags=["audit"],
        posterior_label_mode="benefit",
        use_benefit_sample_weights=False,
        state_scope="all",
        repair_mode="full",
    )
    source_count = clean.shape[0]
    adv_candidate = detector_bundle.rec_inputs[:source_count]
    adv_observed = detector_bundle.obs_inputs[:source_count]
    if np.array_equal(adv_candidate, adv_observed):
        raise AssertionError("Posterior detector full-state candidate unexpectedly equals observed state.")
    metadata = dict(detector_bundle.metadata or {})
    if metadata.get("repair_mode") != "full":
        raise AssertionError("Posterior detector dataset did not record full repair mode.")
    if metadata.get("posterior_candidate_state") != "full_reconstruction":
        raise AssertionError("Posterior detector candidate metadata is not runtime-aligned.")
    return {
        "repair_mode": metadata.get("repair_mode"),
        "posterior_candidate_state": metadata.get("posterior_candidate_state"),
        "repaired_state_dimensions": "all_11",
        "sample_count": int(detector_bundle.obs_inputs.shape[0]),
    }


def audit_multiday_bounds() -> dict:
    result = {}
    for split in ("train", "val"):
        union_low, union_high = union_observation_bounds(split)
        frame = load_manifest(split)
        checked = 0
        for _, row in frame.iloc[:: max(1, len(frame) // 12)].iterrows():
            arrivals, signal_path, _ = load_scenario(row)
            from evc.merged_core import ChargingEnv, TRAIN_PROFILE
            env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
            low, high = env.observation_bounds(max_duration_of_stay=max(12, int(arrivals["Duration_of_stay"].max())))
            if np.any(np.asarray(low) < union_low - 1e-7) or np.any(np.asarray(high) > union_high + 1e-7):
                raise AssertionError(f"Split-level union bounds do not contain a {split} scenario.")
            checked += 1
        result[split] = {
            "checked_scenarios": checked,
            "union_low": union_low.tolist(),
            "union_high": union_high.tolist(),
            "contains_sampled_scenario_bounds": True,
        }
    return result


def audit_orchestration_source() -> dict:
    train_source = (PACKAGE_ROOT / "scripts" / "09_retrain_dtsr_seed42.py").read_text(encoding="utf-8")
    eval_source = (PACKAGE_ROOT / "scripts" / "_strength_eval_common.py").read_text(encoding="utf-8")
    checks = {
        "dae_validation_callback_present": "validator=dae_validator" in train_source,
        "dae_selected_by_full_runtime_score": 'select_by="dae_checkpoint_score"' in train_source,
        "detector_full_labels_train": train_source.count("repair_mode=REPAIR_MODE") >= 2,
        "candidate_comparison_uses_candidate_independent_attack_seed": "candidate_index * 1_000_000" not in train_source,
        "electhacker_c_validation_median_present": "validation_price_median()" in train_source,
        "final_eval_sample_std_ddof1": "np.std(arr, ddof=1)" in eval_source,
        "final_eval_resume_jsonl_present": "table_rollouts.jsonl" in eval_source,
        "adaptive_attack_target_defense_configured": "configure_target_defense(" in eval_source,
        "final_eval_repair_mode_full": "repair_mode=REPAIR_MODE" in eval_source,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Orchestration source audit failed: {failed}")
    return {"checks": checks, "all_pass": True}

def audit_model(actor_path: Path, bundle_path: Path, device: torch.device) -> dict:
    actor = load_actor_from_path(actor_path, device)
    bundle = load_actor_critic_bundle(bundle_path, device)
    assert bundle.get("actor_state_dict") is not None
    assert bundle.get("critic_state_dict") is not None
    assert actor_matches_bundle(actor, bundle), "Actor file does not match actor_state_dict in bundle."
    metadata = dict(bundle.get("metadata") or {})
    checkpoint_episode = int(metadata.get("checkpoint_episode", -1))
    assert checkpoint_episode == 100, f"Expected ep100 model, got {checkpoint_episode}."
    return {
        "actor_path": str(actor_path),
        "bundle_path": str(bundle_path),
        "actor_matches_bundle": True,
        "critic_present": True,
        "checkpoint_episode": checkpoint_episode,
        "validation_summary": metadata.get("validation_summary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DTSR code/data/model integrity before retraining.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--actor-path", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "results" / "dtsr_retrain_seed42" / "audit" / "integrity_audit.json",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = resolve_device(args.device)
    report = {
        "status": "PASS",
        "device": str(device),
        "original_logic_hash_audit": audit_original_logic_hashes(),
        "pipeline_order_audit": audit_pipeline_order(),
        "dataset_audit": audit_dataset(),
        "model_audit": audit_model(args.actor_path, args.bundle_path, device),
        "posterior_runtime_alignment_audit": audit_posterior_runtime_alignment(args.actor_path, device),
        "multiday_bounds_audit": audit_multiday_bounds(),
        "orchestration_source_audit": audit_orchestration_source(),
        "important_note": (
            "The original implementation order is DAE/DET routing -> UG-BCR gate -> Temporal Shield -> Actor. "
            "The paper ablation labels +Shield then +UG-BCR describe module addition order, not runtime call order."
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
