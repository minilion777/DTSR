from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.ug_bcr_v3 import (  # noqa: E402
    ContinuousGateConfig,
    UGBCRV3Config,
    V3_FEATURE_NAMES,
    _v2_config_from_payload,
    reconstruct_v3_features_from_audit,
    ug_bcr_v3_config_payload,
)


ATTACKS = ("clean", "local_deadline_drift_pgd", "local_small_drift_q")
CONSTRAINTS = {
    "clean_activation_max": 0.01,
    "deadline_precision_min": 0.90,
    "deadline_recall_min": 0.65,
    "small_drift_precision_min": 0.65,
    "small_drift_recall_min": 0.55,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float) > 0.5
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).to_numpy(dtype=bool)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    precision = 0.0 if tp + fp == 0 else float(tp / (tp + fp))
    recall = 0.0 if tp + fn == 0 else float(tp / (tp + fn))
    f1 = 0.0 if precision + recall == 0.0 else float(2.0 * precision * recall / (precision + recall))
    return {
        "count": int(len(y)),
        "positives": int(np.sum(y)),
        "activation_rate": float(np.mean(pred)) if len(pred) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _evaluate(frame: pd.DataFrame, y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, dict[str, float | int]]:
    pred = score >= float(threshold)
    if "is_new_arrival" in frame.columns:
        pred &= ~_as_bool(frame["is_new_arrival"])
    attack_values = frame["attack_key"].astype(str).to_numpy()
    return {
        attack: _metrics(y[attack_values == attack], pred[attack_values == attack])
        for attack in ATTACKS
    }


def _constraint_summary(metrics: dict[str, dict[str, float | int]]) -> tuple[bool, float, float]:
    clean = metrics["clean"]
    deadline = metrics["local_deadline_drift_pgd"]
    small = metrics["local_small_drift_q"]
    violations = (
        max(float(clean["activation_rate"]) - CONSTRAINTS["clean_activation_max"], 0.0)
        / CONSTRAINTS["clean_activation_max"]
        + max(CONSTRAINTS["deadline_precision_min"] - float(deadline["precision"]), 0.0)
        / CONSTRAINTS["deadline_precision_min"]
        + max(CONSTRAINTS["deadline_recall_min"] - float(deadline["recall"]), 0.0)
        / CONSTRAINTS["deadline_recall_min"]
        + max(CONSTRAINTS["small_drift_precision_min"] - float(small["precision"]), 0.0)
        / CONSTRAINTS["small_drift_precision_min"]
        + max(CONSTRAINTS["small_drift_recall_min"] - float(small["recall"]), 0.0)
        / CONSTRAINTS["small_drift_recall_min"]
    )
    feasible = bool(violations <= 1e-12)
    # F1 and recall are explicit calibration objectives after satisfying guardrails.
    quality = float(
        0.35 * float(deadline["f1"])
        + 0.35 * float(small["f1"])
        + 0.15 * float(deadline["recall"])
        + 0.15 * float(small["recall"])
    )
    return feasible, float(violations), quality


def _group_balanced_weights(frame: pd.DataFrame, y: np.ndarray, positive_weight: float) -> np.ndarray:
    attacks = frame["attack_key"].astype(str).to_numpy()
    weights = np.zeros(len(frame), dtype=np.float64)
    for attack in ATTACKS:
        mask = attacks == attack
        if np.any(mask):
            weights[mask] = 1.0 / float(np.sum(mask))
    weights *= float(len(weights) / max(np.sum(weights), 1e-12))
    weights[y] *= float(positive_weight)
    return weights


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    l2: float,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y.astype(np.float64), dtype=torch.float64)
    wt = torch.as_tensor(sample_weight, dtype=torch.float64)
    coef = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    prevalence = float(np.average(y.astype(float), weights=sample_weight))
    initial_intercept = float(np.log(np.clip(prevalence, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - prevalence, 1e-5, 1.0)))
    intercept = torch.tensor(initial_intercept, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [coef, intercept],
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = xt.mv(coef) + intercept
        point_loss = torch.clamp(logits, min=0.0) - logits * yt + torch.log1p(torch.exp(-torch.abs(logits)))
        loss = torch.sum(point_loss * wt) / torch.sum(wt) + 0.5 * float(l2) * torch.sum(coef * coef)
        loss.backward()
        return loss

    optimizer.step(closure)
    return coef.detach().cpu().numpy(), float(intercept.detach().cpu())


def _predict(x: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    logits = np.clip(x @ coef + float(intercept), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _flatten_metrics(
    metrics: dict[str, dict[str, float | int]],
    *,
    split: str,
    model_id: str,
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "model_id": model_id,
            "decision_threshold": float(threshold),
            "attack_key": attack,
            **values,
        }
        for attack, values in metrics.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the independent continuous-score UG-BCR v3 gate.")
    parser.add_argument(
        "--validation-audit-path",
        type=Path,
        default=PACKAGE_ROOT
        / "results"
        / "ug_bcr_v3_screening_val_v2audit_20scenes_seed42"
        / "tables"
        / "ug_bcr_gate_quality_raw.csv",
    )
    parser.add_argument(
        "--test-audit-path",
        type=Path,
        default=PACKAGE_ROOT
        / "results"
        / "ug_bcr_gate_quality_seed42_newlong_v2_newcal"
        / "tables"
        / "ug_bcr_gate_quality_raw.csv",
    )
    parser.add_argument(
        "--base-v2-config-path",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "ug_bcr_seed42_fullstate_newlong_v2" / "ug_bcr_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "ug_bcr_v3_seed42_newlong_v2",
    )
    parser.add_argument("--train-scene-fraction", type=float, default=0.60)
    parser.add_argument("--max-iter", type=int, default=60)
    args = parser.parse_args()

    config_path = args.output_dir / "ug_bcr_v3_config.json"
    if config_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing v3 config: {config_path}")
    for path in (args.validation_audit_path, args.test_audit_path, args.base_v2_config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = reconstruct_v3_features_from_audit(pd.read_csv(args.validation_audit_path))
    test = reconstruct_v3_features_from_audit(pd.read_csv(args.test_audit_path))
    validation = validation[validation["attack_key"].astype(str).isin(ATTACKS)].copy()
    test = test[test["attack_key"].astype(str).isin(ATTACKS)].copy()
    scenes = sorted(validation["scenario_id"].astype(str).unique().tolist())
    train_count = max(1, min(len(scenes) - 1, int(round(len(scenes) * float(args.train_scene_fraction)))))
    train_scenes = set(scenes[:train_count])
    train_mask = validation["scenario_id"].astype(str).isin(train_scenes).to_numpy()
    calibration_mask = ~train_mask

    x_all = validation.loc[:, list(V3_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y_all = _as_bool(validation["oracle_positive"])
    means = np.nanmean(x_all[train_mask], axis=0)
    scales = np.nanstd(x_all[train_mask], axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    x_all = np.nan_to_num((x_all - means) / scales, nan=0.0, posinf=0.0, neginf=0.0)
    x_train, y_train = x_all[train_mask], y_all[train_mask]
    calibration_frame = validation.loc[calibration_mask].copy()
    y_calibration = y_all[calibration_mask]

    thresholds = np.linspace(0.005, 0.995, 199)
    candidate_rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[np.ndarray, float, float, float]] = {}
    for positive_weight in (0.75, 1.0, 1.5, 2.0, 3.0):
        for l2 in (1e-4, 1e-3, 1e-2):
            model_id = f"pw{positive_weight:g}_l2{l2:g}"
            weights = _group_balanced_weights(validation.loc[train_mask], y_train, positive_weight)
            coef, intercept = _fit_logistic(x_train, y_train, weights, l2=l2, max_iter=int(args.max_iter))
            fitted[model_id] = (coef, intercept, positive_weight, l2)
            score = _predict(x_all[calibration_mask], coef, intercept)
            best_row: dict[str, Any] | None = None
            best_rank: tuple[int, float, float, float] | None = None
            for threshold in thresholds:
                metrics = _evaluate(calibration_frame, y_calibration, score, float(threshold))
                feasible, violation, quality = _constraint_summary(metrics)
                row = {
                    "model_id": model_id,
                    "positive_weight": positive_weight,
                    "l2": l2,
                    "decision_threshold": float(threshold),
                    "feasible": feasible,
                    "constraint_violation": violation,
                    "calibration_objective": quality,
                    "clean_activation": metrics["clean"]["activation_rate"],
                    "deadline_precision": metrics["local_deadline_drift_pgd"]["precision"],
                    "deadline_recall": metrics["local_deadline_drift_pgd"]["recall"],
                    "deadline_f1": metrics["local_deadline_drift_pgd"]["f1"],
                    "small_drift_precision": metrics["local_small_drift_q"]["precision"],
                    "small_drift_recall": metrics["local_small_drift_q"]["recall"],
                    "small_drift_f1": metrics["local_small_drift_q"]["f1"],
                }
                rank = (int(feasible), -float(violation), float(quality), -float(threshold))
                if best_rank is None or rank > best_rank:
                    best_row = row
                    best_rank = rank
            assert best_row is not None
            candidate_rows.append(best_row)
            print(
                f"{model_id}: feasible={best_row['feasible']} threshold={best_row['decision_threshold']:.3f} "
                f"clean={100*best_row['clean_activation']:.2f}% "
                f"deadline P/R/F1={100*best_row['deadline_precision']:.1f}/{100*best_row['deadline_recall']:.1f}/{100*best_row['deadline_f1']:.1f} "
                f"small P/R/F1={100*best_row['small_drift_precision']:.1f}/{100*best_row['small_drift_recall']:.1f}/{100*best_row['small_drift_f1']:.1f}",
                flush=True,
            )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["feasible", "constraint_violation", "calibration_objective"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    selected = candidates.iloc[0].to_dict()
    selected_id = str(selected["model_id"])
    coef, intercept, positive_weight, l2 = fitted[selected_id]
    threshold = float(selected["decision_threshold"])
    base_payload = json.loads(args.base_v2_config_path.read_text(encoding="utf-8"))
    config = UGBCRV3Config(
        base_v2=_v2_config_from_payload(base_payload),
        continuous_gate=ContinuousGateConfig(
            feature_means=tuple(float(value) for value in means),
            feature_scales=tuple(float(value) for value in scales),
            coefficients=tuple(float(value) for value in coef),
            intercept=float(intercept),
            decision_threshold=threshold,
        ),
        training_metadata={
            "seed": 42,
            "validation_audit_path": str(args.validation_audit_path),
            "test_audit_path_reporting_only": str(args.test_audit_path),
            "base_v2_config_path": str(args.base_v2_config_path),
            "belief_estimator_frozen": True,
            "gate_model": "weighted_logistic_regression",
            "train_scenes": sorted(train_scenes),
            "calibration_scenes": sorted(set(scenes) - train_scenes),
            "positive_weight": float(positive_weight),
            "l2": float(l2),
            "selection_uses_test": False,
            "constraints": CONSTRAINTS,
            "calibration_objective": "0.35 deadline_F1 + 0.35 small_F1 + 0.15 deadline_recall + 0.15 small_recall",
            "offline_constraints_feasible": bool(selected["feasible"]),
            "closed_loop_recovery_guardrail": "pending: each attack recovery >= current_v2 - 2 percentage points",
        },
    )
    _write_json(config_path, ug_bcr_v3_config_payload(config))
    candidates.to_csv(args.output_dir / "candidate_summary.csv", index=False, encoding="utf-8-sig")

    score_validation = _predict(x_all, coef, intercept)
    validation_rows: list[dict[str, Any]] = []
    for split_name, mask in (("train", train_mask), ("calibration", calibration_mask), ("all_validation", np.ones(len(validation), dtype=bool))):
        metrics = _evaluate(validation.loc[mask], y_all[mask], score_validation[mask], threshold)
        validation_rows.extend(_flatten_metrics(metrics, split=split_name, model_id=selected_id, threshold=threshold))
    pd.DataFrame(validation_rows).to_csv(args.output_dir / "offline_validation_metrics.csv", index=False, encoding="utf-8-sig")

    x_test = test.loc[:, list(V3_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    x_test = np.nan_to_num((x_test - means) / scales, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = _as_bool(test["oracle_positive"])
    test_metrics = _evaluate(test, y_test, _predict(x_test, coef, intercept), threshold)
    pd.DataFrame(_flatten_metrics(test_metrics, split="test_reporting_only", model_id=selected_id, threshold=threshold)).to_csv(
        args.output_dir / "offline_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    _write_json(
        args.output_dir / "manifest.json",
        {
            "selected_model": selected,
            "config": str(config_path),
            "offline_gate_constraints_checked": True,
            "closed_loop_recovery_checked": False,
            "note": "Test audit was excluded from fitting and threshold selection.",
        },
    )
    print(f"Selected {selected_id}, threshold={threshold:.3f}, feasible={bool(selected['feasible'])}")
    print(f"Saved independent v3 artifact: {config_path}")


if __name__ == "__main__":
    main()
