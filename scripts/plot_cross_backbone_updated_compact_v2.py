from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


TEXT_COLOR = "#262626"
GRID_COLOR = "#E9E9E9"

ATTACK_COLORS = {
    "PGD": "#5E92AF",
    "FGSM": "#8DB9CC",
    "Q-function": "#9A8CC7",
    "ElectHacker-C": "#4F9A77",
    "ElectHacker-F": "#72B7A1",
    "ElectHacker-O": "#A0C98D",
    "Small-drift Q": "#D2A74B",
    "Deadline-PGD": "#C48A6A",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "axes.edgecolor": TEXT_COLOR,
            "axes.linewidth": 0.72,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_interval(ax, mean: float, low: float, high: float, y: float, color: str, height: float = 0.20) -> None:
    ax.add_patch(
        Rectangle(
            (low, y - height / 2.0),
            high - low,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.35,
            alpha=0.84,
            zorder=2,
        )
    )
    ax.vlines(
        mean,
        y - height * 0.40,
        y + height * 0.40,
        linewidth=0.55,
        color="#333333",
        zorder=3,
    )


def style_axis(ax) -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.76)
    ax.tick_params(axis="x", length=2.4, width=0.68)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.42, zorder=0)
    ax.grid(axis="y", visible=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-csv", type=Path, required=True)
    parser.add_argument("--clean-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = pd.read_csv(args.plot_csv)
    clean_df = pd.read_csv(args.clean_csv)

    algorithms = plot_df["algorithm"].drop_duplicates().tolist()
    attacks = plot_df["attack"].drop_duplicates().tolist()

    plot_df["attack"] = pd.Categorical(plot_df["attack"], categories=attacks, ordered=True)
    plot_df["algorithm"] = pd.Categorical(plot_df["algorithm"], categories=algorithms, ordered=True)
    plot_df = plot_df.sort_values(["algorithm", "attack"]).reset_index(drop=True)

    clean_by_algorithm = {str(row["Algorithm"]): float(row["Δ return"]) for _, row in clean_df.iterrows()}

    # Use a compact but readable attack-row spacing and recompute a symmetric
    # vertical range so the rows do not float inside excessive blank space.
    original_spacing = 0.52
    spacing = original_spacing * 0.56
    compact_span = (len(attacks) - 1) * spacing
    y_positions = {
        attack: (len(attacks) - 1 - i) * spacing
        for i, attack in enumerate(attacks)
    }
    yticks = [y_positions[a] for a in attacks]
    vertical_padding = 0.24
    ymax = compact_span + vertical_padding
    ymin = -vertical_padding

    color_map = {attack: ATTACK_COLORS.get(attack, "#999999") for attack in attacks}

    # Standard two-column text width with enough height for eight attack rows.
    fig = plt.figure(figsize=(7.16, 2.55), facecolor="white")

    # Build the recovery and clean-state panels as two independent figures,
    # then stack them. Their internal label spacing is therefore decoupled
    # from the gap between the two panel groups.
    top_figure, bottom_figure = fig.subfigures(
        2,
        1,
        height_ratios=(0.62, 0.38),
        hspace=0.015,
    )
    top_axes = top_figure.subplots(1, len(algorithms), squeeze=False)[0]
    bottom_axes = bottom_figure.subplots(1, len(algorithms), squeeze=False)[0]
    top_figure.subplots_adjust(left=0.14, right=0.985, top=0.90, bottom=0.22, wspace=0.10)
    bottom_figure.subplots_adjust(left=0.14, right=0.985, top=0.94, bottom=0.12, wspace=0.10)
    top_figure.supxlabel("(a) Attack recovery rate (%)", x=0.5625, y=0.03, fontsize=8.0)
    bottom_figure.supxlabel("(b) Clean-state Δ return", x=0.5625, y=0.30, fontsize=8.0)

    audit_rows = []

    for col, algorithm in enumerate(algorithms):
        top_ax = top_axes[col]
        bottom_ax = bottom_axes[col]

        sub = plot_df[plot_df["algorithm"] == algorithm]

        for _, row in sub.iterrows():
            attack = str(row["attack"])
            draw_interval(
                top_ax,
                mean=float(row["recovery_mean_pct"]),
                low=float(row["ci95_low_pct"]),
                high=float(row["ci95_high_pct"]),
                y=y_positions[attack],
                color=color_map[attack],
            )
            audit_rows.append(
                [
                    algorithm,
                    attack,
                    float(row["recovery_mean_pct"]),
                    float(row["ci95_low_pct"]),
                    float(row["ci95_high_pct"]),
                    clean_by_algorithm.get(algorithm, float("nan")),
                ]
            )

        top_ax.set_xlim(75, 100.3)
        top_ax.set_xticks([75, 80, 85, 90, 95, 100])
        top_ax.set_ylim(ymin, ymax)
        top_ax.set_title(algorithm, pad=3.0, fontweight="bold")
        top_ax.set_yticks(yticks)

        if col == 0:
            top_ax.set_yticklabels(attacks, fontsize=7.0)
            top_ax.tick_params(axis="y", length=0, pad=3.8)
        else:
            top_ax.set_yticklabels([])
            top_ax.tick_params(axis="y", length=0)

        style_axis(top_ax)
        top_ax.tick_params(axis="x", pad=1.5)

        delta = clean_by_algorithm.get(algorithm, float("nan"))
        bottom_ax.scatter([delta], [0], s=24, zorder=4)
        bottom_ax.annotate(
            f"{delta:.1f}",
            (delta, 0),
            xytext=(0, 3.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#404040",
        )
        bottom_ax.set_xlim(-10, 0.3)
        bottom_ax.set_xticks([-10, -5, 0])
        bottom_ax.set_ylim(-0.20, 0.10)
        bottom_ax.set_yticks([])
        style_axis(bottom_ax)
        bottom_ax.grid(axis="x", visible=False)
        bottom_ax.vlines(
            [-10, -5, 0],
            0,
            0.10,
            color=GRID_COLOR,
            linewidth=0.42,
            zorder=0,
        )
        bottom_ax.spines["bottom"].set_position(("data", 0))

    stem = args.output_dir / "figure_cross_backbone_updated_compact_v2"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)

    with stem.with_name(stem.name + "_values.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Attack", "Recovery mean (%)", "CI95 low (%)", "CI95 high (%)", "Clean-state delta return"])
        writer.writerows(audit_rows)

    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
