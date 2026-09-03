from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from _common import PACKAGE_ROOT, write_json

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.native_dtsr import default_native_bundle_path, native_artifact_layout


STAGES = ("collect", "dae", "det", "shield", "ug_bcr", "evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the backbone-native TD3/SAC/PPO DTSR pipeline with isolated artifacts."
    )
    parser.add_argument("--algorithm", choices=["td3", "sac", "ppo"], required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument(
        "--native-config",
        type=Path,
        default=PACKAGE_ROOT / "results" / "native_attack_calibration_seed42" / "native_attack_config.json",
    )
    parser.add_argument("--from-stage", choices=STAGES, default="collect")
    parser.add_argument("--through-stage", choices=STAGES, default="evaluate")
    parser.add_argument("--train-scenes", type=int, default=200)
    parser.add_argument("--val-scenes", type=int, default=40)
    parser.add_argument("--dae-epochs", type=int, default=50)
    parser.add_argument("--detector-epochs", type=int, default=30)
    parser.add_argument("--shield-clean-scenes", type=int, default=20)
    parser.add_argument("--shield-tuning-scenes", type=int, default=5)
    parser.add_argument("--ug-stage1-scenes", type=int, default=4)
    parser.add_argument("--ug-stage2-scenes", type=int, default=8)
    parser.add_argument("--test-scenes", type=int, default=20)
    parser.add_argument(
        "--test-selection-mode",
        choices=("seeded", "first"),
        default="seeded",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    first = STAGES.index(args.from_stage)
    last = STAGES.index(args.through_stage)
    if first > last:
        raise ValueError("--from-stage must not come after --through-stage.")
    selected_stages = STAGES[first : last + 1]
    bundle_path = args.bundle_path or default_native_bundle_path(
        PACKAGE_ROOT, args.algorithm, args.seed
    )
    common = [
        "--algorithm", args.algorithm,
        "--device", args.device,
        "--seed", str(args.seed),
        "--bundle-path", str(bundle_path),
    ]
    with_attacks = [*common, "--native-config", str(args.native_config)]
    commands: dict[str, list[str]] = {
        "collect": [
            sys.executable,
            "scripts/02_collect_multiday_clean.py",
            *common,
            "--train-scenes", str(args.train_scenes),
            "--val-scenes", str(args.val_scenes),
        ],
        "dae": [
            sys.executable,
            "scripts/09_train_dae_only_seed42.py",
            *with_attacks,
            "--dae-epochs", str(args.dae_epochs),
            "--save-pair-datasets",
        ],
        "det": [
            sys.executable,
            "scripts/13_train_det_only_seed42.py",
            *with_attacks,
            "--detector-epochs", str(args.detector_epochs),
            "--save-detector-datasets",
        ],
        "shield": [
            sys.executable,
            "scripts/14_calibrate_temporal_shield_seed42.py",
            *with_attacks,
            "--clean-calibration-scenes", str(args.shield_clean_scenes),
            "--tuning-scenes", str(args.shield_tuning_scenes),
        ],
        "ug_bcr": [
            sys.executable,
            "scripts/15_calibrate_ug_bcr_seed42.py",
            *with_attacks,
            "--stage1-scenes", str(args.ug_stage1_scenes),
            "--stage2-scenes", str(args.ug_stage2_scenes),
        ],
        "evaluate": [
            sys.executable,
            "scripts/31_evaluate_native_dtsr_backbone.py",
            *with_attacks,
            "--scenes", str(args.test_scenes),
            "--selection-mode", args.test_selection_mode,
        ],
    }

    layout = native_artifact_layout(PACKAGE_ROOT, args.algorithm, args.seed)
    record_path = layout["dtsr_results"] / "pipeline_run.json"
    started = time.perf_counter()
    completed: list[str] = []
    for stage in selected_stages:
        command = commands[stage]
        print("+", subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PACKAGE_ROOT, check=True)
        completed.append(stage)

    record = {
        "status": "dry_run" if args.dry_run else "complete",
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "bundle_path": str(Path(bundle_path).resolve()),
        "native_config": str(Path(args.native_config).resolve()),
        "selected_stages": list(selected_stages),
        "completed_stages": completed,
        "commands": {stage: commands[stage] for stage in selected_stages},
        "elapsed_minutes": float((time.perf_counter() - started) / 60.0),
    }
    if not args.dry_run:
        write_json(record_path, record)
    print(f"[Native DTSR Pipeline] {record['status']}: {args.algorithm} {list(selected_stages)}")


if __name__ == "__main__":
    main()
