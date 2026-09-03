"""Render the 120-scenario degradation curves at single-column size."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT / "runs" / "attack_120" / "attack_episode_metrics_long.csv"
DEFAULT_OUTPUT_DIR = PROJECT / "runs" / "figures"

# Full single-column width (3.54 in / 89.9 mm) with a compact height.
FIGURE_SIZE_IN = (3.54, 2.25)
EXPECTED_SCENARIOS = 120

ATTACKS = [
    ("Opposite-PGD", "PGD", "#D55E00", "-"),
    ("Opposite-FGSM", "FGSM", "#E69F00", "--"),
    ("Q-function", "Q-function", "#009E73", "-."),
    ("ElectHacker-C", "EH-C", "#7E57C2", "-"),
    ("ElectHacker-F", "EH-F", "#8C564B", "--"),
    ("ElectHacker-O", "EH-O", "#CC79A7", "-."),
]


def load_degradation_data(source: Path) -> pd.DataFrame:
    """Pair every attacked run with its clean run and compute degradation."""
    frame = pd.read_csv(source)
    required = {"scenario_id", "episode", "condition", "ep_reward"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean = frame.loc[
        frame["condition"].eq("Clean"),
        ["scenario_id", "episode", "ep_reward"],
    ].rename(
        columns={"episode": "clean_episode", "ep_reward": "clean_reward"}
    )
    if clean["scenario_id"].duplicated().any():
        raise ValueError("Clean runs must contain one row per scenario_id")

    paired = frame.merge(clean, on="scenario_id", how="left", validate="many_to_one")
    paired["plot_episode"] = paired["episode"].fillna(paired["clean_episode"])
    paired["degradation"] = paired["clean_reward"] - paired["ep_reward"]

    for condition, *_ in ATTACKS:
        subset = paired.loc[paired["condition"].eq(condition)]
        if len(subset) != EXPECTED_SCENARIOS:
            raise ValueError(
                f"Expected {EXPECTED_SCENARIOS} rows for {condition}, got {len(subset)}"
            )
        if subset[["plot_episode", "degradation"]].isna().any().any():
            raise ValueError(f"Incomplete paired data for {condition}")

    return paired


def render_figure(paired: pd.DataFrame, output_stem: Path) -> None:
    """Draw the original six-curve form with compact journal typography."""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 6.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN)

    for condition, label, color, linestyle in ATTACKS:
        values = paired.loc[paired["condition"].eq(condition)].sort_values(
            "plot_episode"
        )
        ax.plot(
            values["plot_episode"],
            values["degradation"],
            color=color,
            linestyle=linestyle,
            linewidth=0.8,
            label=label,
            solid_capstyle="round",
        )

    ax.set_xlim(1, EXPECTED_SCENARIOS)
    ax.set_ylim(0, 4000)
    ax.set_xticks([1, 30, 60, 90, 120])
    ax.set_yticks(np.arange(0, 4001, 1000))
    ax.set_xlabel("Test scenario index", labelpad=2.0)
    ax.set_ylabel("Reward degradation", labelpad=1.2)

    # Use only light solid guides above zero. The y=0 reference line is omitted.
    for y_value in (1000, 2000, 3000, 4000):
        ax.axhline(
            y_value,
            color="#D9D9D9",
            linewidth=0.4,
            linestyle="-",
            zorder=0,
        )

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.65, length=2.8, pad=1.2)
    x_tick_labels = ax.get_xticklabels()
    x_tick_labels[0].set_horizontalalignment("left")
    x_tick_labels[-1].set_horizontalalignment("right")

    # The legend occupies the naturally empty region below all six curves.
    handles, legend_labels = ax.get_legend_handles_labels()
    # Matplotlib fills a multi-column legend down each column. Interleave the
    # handles so the first row contains the three conventional attacks and the
    # second row contains the three task-oriented EH attacks.
    legend_order = [0, 3, 1, 4, 2, 5]
    legend = ax.legend(
        [handles[index] for index in legend_order],
        [legend_labels[index] for index in legend_order],
        loc="lower right",
        bbox_to_anchor=(0.995, 0.025),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#A8A8A8",
        handlelength=2.5,
        handletextpad=0.5,
        columnspacing=0.8,
        borderaxespad=0.0,
        borderpad=0.45,
    )
    legend.get_frame().set_linewidth(0.5)

    fig.subplots_adjust(left=0.115, right=0.995, top=0.985, bottom=0.17)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_stem}.png", dpi=600, facecolor="white")
    fig.savefig(f"{output_stem}.pdf", facecolor="white")
    fig.savefig(f"{output_stem}.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the 120-scenario attack-degradation figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="attack_degradation_120_single_column")
    args = parser.parse_args()
    output_stem = args.output_dir / args.output_name
    paired = load_degradation_data(args.input)
    render_figure(paired, output_stem)
    for suffix in ("png", "pdf", "svg"):
        print(f"Saved: {output_stem}.{suffix}")


if __name__ == "__main__":
    main()
