from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DET_SUMMARY = (
    PACKAGE_ROOT
    / "results"
    / "exp4_attack_ratio_20scenes_seed42"
    / "tables"
    / "exp4_det_routing_quality_summary.csv"
)
DEFAULT_UG_SUMMARY = (
    PACKAGE_ROOT
    / "results"
    / "ug_bcr_gate_quality_20scenes_seed42"
    / "tables"
    / "ug_bcr_gate_quality_summary.csv"
)
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "results" / "gate_quality_figures"
DEFAULT_LATEST_DIR = PACKAGE_ROOT / "results" / "paper_tables_latest"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_all(fig: plt.Figure, output_dir: Path, stem: str, latest_dir: Path | None = None, latest_stem: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
        if latest_dir is not None:
            latest_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(latest_dir / f"{latest_stem or stem}.{suffix}", bbox_inches="tight")


def metric_matrix(frame: pd.DataFrame, metric: str) -> tuple[list[str], list[str], np.ndarray]:
    rows = list(dict.fromkeys(frame["attack"].astype(str).tolist()))
    ratios = sorted(float(value) for value in frame["attack_ratio"].unique())
    columns = [f"{ratio:.2f}" for ratio in ratios]
    matrix = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for r_idx, attack in enumerate(rows):
        for c_idx, ratio in enumerate(ratios):
            subset = frame[(frame["attack"].astype(str) == attack) & (np.isclose(frame["attack_ratio"].astype(float), ratio))]
            if not subset.empty:
                matrix[r_idx, c_idx] = float(subset.iloc[0][metric])
    return rows, columns, matrix


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
) -> None:
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Attack coverage ratio rho")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            text_color = "white" if value > (vmin + vmax) / 2.0 else "#1F1F1F"
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=7.8)
    ax.tick_params(length=0)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=12)


def plot_det_heatmaps(det_summary_path: Path, output_dir: Path, latest_dir: Path | None) -> Path:
    frame = pd.read_csv(det_summary_path)
    required = {
        "attack",
        "attack_ratio",
        "benefit_routing_f1_pct_mean",
        "net_benefit_capture_rate_pct_mean",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"DET summary is missing columns: {missing}")

    row_labels, col_labels, f1 = metric_matrix(frame, "benefit_routing_f1_pct_mean")
    _, _, nbcr = metric_matrix(frame, "net_benefit_capture_rate_pct_mean")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    draw_heatmap(
        axes[0],
        f1,
        row_labels,
        col_labels,
        title="DET routing F1",
        cmap="YlGnBu",
        vmin=75,
        vmax=100,
        colorbar_label="F1 (%)",
    )
    draw_heatmap(
        axes[1],
        nbcr,
        row_labels,
        col_labels,
        title="Net benefit capture",
        cmap="YlOrBr",
        vmin=90,
        vmax=100,
        colorbar_label="NBCR (%)",
    )
    fig.suptitle("Decision-aware DET routing under mixed short-horizon attacks", y=1.06, fontsize=10.5)
    save_all(
        fig,
        output_dir,
        "figure_det_routing_quality_heatmaps",
        latest_dir,
        "figure5_det_routing_quality_latest",
    )
    plt.close(fig)
    return output_dir / "figure_det_routing_quality_heatmaps.png"


def plot_ug_bcr_bars(ug_summary_path: Path, output_dir: Path, latest_dir: Path | None) -> Path:
    frame = pd.read_csv(ug_summary_path)
    required = {
        "attack_display_name",
        "attack_family",
        "ug_precision_pct",
        "ug_recall_pct",
        "ug_f1_pct",
        "ug_activation_rate_pct",
        "improvement_capture_pct",
        "ug_precision_scene_std_pct",
        "ug_recall_scene_std_pct",
        "ug_f1_scene_std_pct",
        "ug_activation_scene_std_pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"UG-BCR summary is missing columns: {missing}")

    long_frame = frame[frame["attack_family"].astype(str) == "long"].copy()
    if long_frame.empty:
        raise ValueError("UG-BCR summary has no long-horizon attack rows.")

    display_order = ["deadline_pgd", "small_drift_q"]
    paper_labels = {
        "Clean": "Clean",
        "deadline_pgd": "Deadline-PGD",
        "small_drift_q": "Small-drift Q",
    }
    long_frame["_order"] = long_frame["attack_display_name"].astype(str).map(
        {name: idx for idx, name in enumerate(display_order)}
    ).fillna(99)
    long_frame = long_frame.sort_values(["_order", "attack_display_name"], kind="mergesort").drop(columns=["_order"])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)

    metrics = [
        ("Precision", "ug_precision_pct", "ug_precision_scene_std_pct", "#4E79A7"),
        ("Recall", "ug_recall_pct", "ug_recall_scene_std_pct", "#F28E2B"),
        ("F1", "ug_f1_pct", "ug_f1_scene_std_pct", "#59A14F"),
    ]
    x = np.arange(len(long_frame))
    width = 0.24
    for idx, (label, mean_col, std_col, color) in enumerate(metrics):
        values = long_frame[mean_col].astype(float).to_numpy()
        errors = long_frame[std_col].astype(float).to_numpy()
        offsets = x + (idx - 1) * width
        axes[0].bar(offsets, values, width=width, color=color, label=label, yerr=errors, capsize=2.5, linewidth=0)
        for xpos, value in zip(offsets, values):
            axes[0].text(xpos, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_title("UG-BCR gate classification", pad=34)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        long_frame["attack_display_name"].astype(str).map(paper_labels).fillna(
            long_frame["attack_display_name"].astype(str)
        ).tolist()
    )
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
    )

    usage_frame = frame.copy()
    usage_frame["_order"] = usage_frame["attack_display_name"].astype(str).map(
        {"Clean": 0, "deadline_pgd": 1, "small_drift_q": 2}
    ).fillna(99)
    usage_frame = usage_frame.sort_values(["_order", "attack_display_name"], kind="mergesort").drop(columns=["_order"])
    labels = usage_frame["attack_display_name"].astype(str).map(paper_labels).fillna(
        usage_frame["attack_display_name"].astype(str)
    ).tolist()
    x2 = np.arange(len(usage_frame))
    activation = usage_frame["ug_activation_rate_pct"].astype(float).to_numpy()
    activation_std = usage_frame["ug_activation_scene_std_pct"].astype(float).to_numpy()
    capture = usage_frame["improvement_capture_pct"].astype(float).to_numpy()

    axes[1].bar(
        x2,
        activation,
        width=0.45,
        color="#76B7B2",
        yerr=activation_std,
        capsize=2.5,
        label="Activation",
        linewidth=0,
    )
    long_mask = usage_frame["attack_family"].astype(str).to_numpy() == "long"
    axes[1].scatter(
        x2[long_mask],
        capture[long_mask],
        marker="D",
        s=42,
        color="#E15759",
        label="Improve capture",
        zorder=3,
    )
    capture_by_x = {
        int(xpos): float(value) for xpos, value in zip(x2[long_mask], capture[long_mask])
    }
    for xpos, value in zip(x2, activation):
        nearby_capture = capture_by_x.get(int(xpos))
        overlaps_capture = nearby_capture is not None and abs(float(value) - nearby_capture) < 7.0
        axes[1].text(
            xpos,
            value - 2.0 if overlaps_capture else value + 2.0,
            f"{value:.1f}",
            ha="center",
            va="top" if overlaps_capture else "bottom",
            fontsize=7,
            color="white" if overlaps_capture else "black",
        )
    for xpos, value in zip(x2[long_mask], capture[long_mask]):
        axes[1].text(xpos, value + 3.0, f"{value:.1f}", ha="center", va="bottom", fontsize=7, color="#B33B3D")
    axes[1].set_title("Gate usage and captured benefit", pad=34)
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )

    fig.suptitle("UG-BCR belief gate quality under long-horizon drift attacks", y=1.08, fontsize=10.5)
    save_all(
        fig,
        output_dir,
        "figure_ug_bcr_gate_quality",
        latest_dir,
        "figure6_ug_bcr_gate_quality_latest",
    )
    plt.close(fig)
    return output_dir / "figure_ug_bcr_gate_quality.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DET and UG-BCR gate-quality figures with matplotlib.")
    parser.add_argument("--det-summary", type=Path, default=DEFAULT_DET_SUMMARY)
    parser.add_argument("--ug-summary", type=Path, default=DEFAULT_UG_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latest-dir", type=Path, default=DEFAULT_LATEST_DIR)
    parser.add_argument("--skip-latest", action="store_true")
    args = parser.parse_args()

    configure_matplotlib()
    latest_dir = None if args.skip_latest else args.latest_dir
    det_path = plot_det_heatmaps(args.det_summary, args.output_dir, latest_dir)
    ug_path = plot_ug_bcr_bars(args.ug_summary, args.output_dir, latest_dir)
    print(f"Saved DET figure: {det_path}")
    print(f"Saved UG-BCR figure: {ug_path}")


if __name__ == "__main__":
    main()
