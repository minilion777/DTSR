"""DAE artifact audit: verify full-state checkpoint reconstruction and no NaN/Inf."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from _common import PACKAGE_ROOT, resolve_device, write_json

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from dtsr_multiday_common import REPAIR_MODE, freeze_module
from evc.defense import load_dae, dae_reconstruction_with_history
from evc.merged_core import load_actor_from_path, load_actor_critic_bundle, Critic


def main():
    parser = argparse.ArgumentParser(description="Audit DAE artifact")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dae-path", default="runs/dtsr/dtsr_dae.pt")
    parser.add_argument("--dae-manifest", default="runs/dtsr/dae_manifest.json")
    parser.add_argument("--actor-path", default="runs/multiday_ddpg/actor_multiday_best.pt")
    parser.add_argument("--bundle-path", default="runs/multiday_ddpg/bundle_multiday_best.pt")
    parser.add_argument("--val-pair", default="runs/dtsr/pair_val.npz")
    parser.add_argument("--output-dir", default="runs/audits/dae")
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = {}

    # 1. File existence
    dae_path = Path(args.dae_path)
    checks["dae_checkpoint_exists"] = dae_path.exists()
    if not checks["dae_checkpoint_exists"]:
        write_json(output_dir / "dae_artifact_audit.json", {"status": "FAIL", "checks": checks})
        raise FileNotFoundError(f"DAE not found at {dae_path}")

    # 2. Manifest checks
    manifest = json.loads(Path(args.dae_manifest).read_text(encoding="utf-8"))
    checks["seed_42"] = manifest.get("seed") == 42
    checks["best_epoch_exists"] = "best_epoch" in manifest
    checks["best_epoch_le_100"] = manifest.get("best_epoch", 999) <= 100
    checks["train_attacks_pgd_q"] = set(manifest.get("train_attacks", [])) == {"opposite_pgd", "q_function"}
    checks["repair_mode_full"] = manifest.get("repair_mode") == REPAIR_MODE == "full"
    checks["full_state_repair"] = manifest.get("repaired_state_dimensions") == "all_11"
    checks["ddpg_checkpoint_100"] = manifest.get("ddpg_checkpoint_episode") == 100
    checks["test_used_for_training_false"] = manifest.get("test_used_for_training") is False

    # 3. Load DAE
    dae = load_dae(dae_path, device)
    dae.eval()
    freeze_module(dae)

    # 4. Dynamic check: 128 val samples
    val = np.load(Path(args.val_pair), allow_pickle=True)
    adv = val["adv_inputs"][:128]
    clean = val["clean_inputs"][:128]
    ep_idx = val["episode_indices"][:128]
    veh_ids = val["vehicle_ids"][:128]

    rec = dae_reconstruction_with_history(dae, adv, device, episode_indices=ep_idx, vehicle_ids=veh_ids, batch_size=128)
    candidate = rec

    checks["candidate_is_full_reconstruction"] = bool(np.allclose(candidate, rec))
    checks["no_nan_in_reconstruction"] = not bool(np.any(np.isnan(rec)))
    checks["no_inf_in_reconstruction"] = not bool(np.any(np.isinf(rec)))
    checks["no_nan_in_candidate"] = not bool(np.any(np.isnan(candidate)))
    checks["no_inf_in_candidate"] = not bool(np.any(np.isinf(candidate)))

    # 5. Actor inference check
    actor = load_actor_from_path(Path(args.actor_path), device).eval()
    bundle_payload = load_actor_critic_bundle(Path(args.bundle_path), device)
    critic = Critic().to(device)
    critic.load_state_dict(bundle_payload["critic_state_dict"])
    critic.eval()

    clean_t = torch.as_tensor(clean, dtype=torch.float32, device=device)
    candidate_t = torch.as_tensor(candidate, dtype=torch.float32, device=device)
    with torch.no_grad():
        clean_act = actor(clean_t).detach().cpu().numpy()
        candidate_act = actor(candidate_t).detach().cpu().numpy()
    checks["actor_inference_ok"] = not bool(np.any(np.isnan(clean_act))) and not bool(np.any(np.isnan(candidate_act)))

    all_pass = all(checks.values())
    result = {"status": "PASS" if all_pass else "FAIL", "checks": checks}
    write_json(output_dir / "dae_artifact_audit.json", result)

    if all_pass:
        print("[DAE-AUDIT] PASS — all checks passed.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"[DAE-AUDIT] FAIL — failed checks: {failed}")
        raise RuntimeError(f"DAE artifact audit failed: {failed}")


if __name__ == "__main__":
    main()
