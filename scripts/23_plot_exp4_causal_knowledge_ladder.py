"""Create the dual-axis causal K0--K4 knowledge-ladder figure.

The input must be produced by scripts/22_evaluate_exp4_causal_knowledge_ladder_seed42.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.exp4_visualization import KNOWLEDGE_ORDER, _common_deadline_scenarios, build_paired_metrics, load_rollouts


LABELS = {
    "K0": "K0\nActor",
    "K1": "K1\n+DAE",
    "K2": "K2\n+DET",
    "K3": "K3\n+UG-BCR",
    "K4": "K4\nFull DTSR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the causal, leak-free adaptive knowledge ladder.")
    parser.add_argument(
        "--input-dir", type=Path,
        default=PACKAGE_ROOT / "results" / "exp4_causal_knowledge_ladder_cuda10_seed42",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.input_dir / "intermediate" / "exp4_rollouts.jsonl"
    protocol_path = args.input_dir / "causal_knowledge_protocol.json"
    if not raw_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("Expected causal rollouts and causal_knowledge_protocol.json in --input-dir")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("attacker") != "causal_knowledge_ladder":
        raise ValueError("Refusing to plot data that is not marked as causal_knowledge_ladder")

    raw = load_rollouts(raw_path)
    paired = build_paired_metrics(raw)
    deadline = _common_deadline_scenarios(paired)
    if deadline["scenario_id"].nunique() == 0:
        raise RuntimeError("No complete paired K0--K4 deadline scenarios are available yet")

    recovery = np.asarray([
        deadline.loc[deadline["knowledge"] == key, "recovery_pct"].mean() for key in KNOWLEDGE_ORDER
    ], dtype=float)
    residual = np.asarray([
        deadline.loc[deadline["knowledge"] == key, "full_exit_rate_pct"].mean() for key in KNOWLEDGE_ORDER
    ], dtype=float)
    x = np.arange(len(KNOWLEDGE_ORDER))
    output_dir = args.output_dir or args.input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 8.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.8,
        "legend.fontsize": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, left = plt.subplots(figsize=(3.54, 2.70), facecolor="white")
    right = left.twinx()
    bars = left.bar(x, recovery, width=0.48, color="#9EB6DA", edgecolor="none", label="Recovery", zorder=2)
    line, = right.plot(
        x, residual, color="#005996", linewidth=1.05, marker="o", markersize=3.2,
        markeredgewidth=0, label="Residual violation", zorder=3,
    )
    left.set_ylabel("Recovery (%)")
    right.set_ylabel("Residual exit violation (%)")
    left.set_xticks(x, [LABELS[key] for key in KNOWLEDGE_ORDER])
    left.set_ylim(max(0.0, float(np.floor(np.nanmin(recovery) / 5.0) * 5.0 - 2.0)), min(100.0, float(np.ceil(np.nanmax(recovery) / 5.0) * 5.0 + 2.0)))
    right.set_ylim(max(0.0, float(np.floor(np.nanmin(residual) / 5.0) * 5.0 - 2.0)), min(100.0, float(np.ceil(np.nanmax(residual) / 5.0) * 5.0 + 2.0)))
    left.spines["top"].set_visible(False)
    left.spines["right"].set_visible(False)
    right.spines["top"].set_visible(False)
    right.spines["left"].set_visible(False)
    left.legend([bars, line], ["Recovery", "Residual violation"], loc="upper center", bbox_to_anchor=(0.52, 1.01), frameon=False, ncol=2)
    fig.subplots_adjust(left=0.16, right=0.86, top=0.93, bottom=0.25)
    stem = output_dir / "fig_exp4_causal_knowledge_ladder"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    caption = (
        f"Causal, leak-free adaptive knowledge ladder under Deadline Denial (n={deadline['scenario_id'].nunique()} paired test scenarios). "
        "K0--K4 add only the declared defense knowledge; no future test signals, oracle routing, Shield leakage at K3, "
        "or post-rollout restart selection is used."
    )
    (output_dir / "fig_exp4_causal_knowledge_ladder_caption.txt").write_text(caption, encoding="utf-8")
    print(f"Saved: {stem.with_suffix('.png')}")
    print(f"Saved: {stem.with_suffix('.pdf')}")
    print(caption)


if __name__ == "__main__":
    main()
