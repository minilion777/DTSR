from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEXT_COLOR = "#262626"
GRID_COLOR = "#E9E9E9"

ALGORITHMS = ["DDPG", "TD3", "SAC", "PPO"]
ATTACK_ORDER = [
    "opposite_pgd",
    "opposite_fgsm",
    "q_function",
    "electhacker_c",
    "electhacker_f",
    "electhacker_o",
    "local_small_drift_q",
    "local_deadline_drift_pgd",
]
ATTACK_LABELS = {
    "opposite_pgd": "PGD",
    "opposite_fgsm": "FGSM",
    "q_function": "Q-function",
    "electhacker_c": "ElectHacker-C",
    "electhacker_f": "ElectHacker-F",
    "electhacker_o": "ElectHacker-O",
    "local_small_drift_q": "Small-drift Q",
    "local_deadline_drift_pgd": "Deadline-PGD",
}
ATTACK_COLORS = {
    "opposite_pgd": "#5E92AF",
    "opposite_fgsm": "#8DB9CC",
    "q_function": "#9A8CC7",
    "electhacker_c": "#4F9A77",
    "electhacker_f": "#72B7A1",
    "electhacker_o": "#A0C98D",
    "local_small_drift_q": "#D2A74B",
    "local_deadline_drift_pgd": "#C48A6A",
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


def _paired_default_from_online_vs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    default_attacks = [
        "opposite_pgd",
        "q_function",
        "local_small_drift_q",
        "local_deadline_drift_pgd",
    ]
    clean = (
        df[(df["method"] == "DDPG") & (df["attack_key"] == "clean")]
        .set_index("scenario_id")["ep_reward"]
        .astype(float)
    )
    rows = []
    for attack in default_attacks:
        raw = (
            df[(df["method"] == "DDPG") & (df["attack_key"] == attack)]
            .set_index("scenario_id")["ep_reward"]
            .astype(float)
        )
        defended = (
            df[(df["method"] == "DTSR") & (df["attack_key"] == attack)]
            .set_index("scenario_id")["ep_reward"]
            .astype(float)
        )
        for scenario_id in sorted(set(clean.index) & set(raw.index) & set(defended.index)):
            attack_degradation = clean.loc[scenario_id] - raw.loc[scenario_id]
            defense_gain = defended.loc[scenario_id] - raw.loc[scenario_id]
            if attack_degradation <= 0:
                continue
            rows.append(
                {
                    "algorithm": "DDPG",
                    "scenario_id": scenario_id,
                    "attack_key": attack,
                    "recovery": defense_gain / attack_degradation,
                    "clean_raw_reward": clean.loc[scenario_id],
                    "attack_raw_reward": raw.loc[scenario_id],
                    "attack_dtsr_reward": defended.loc[scenario_id],
                    "source": str(path),
                }
            )
    return pd.DataFrame(rows)


def _paired_default_from_stage_rollouts(algorithm: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    default_attacks = [
        "opposite_pgd",
        "q_function",
        "local_small_drift_q",
        "local_deadline_drift_pgd",
    ]
    clean = (
        df[(df["attack_key"] == "clean") & (df["stage"] == "attack")]
        .set_index("scenario_id")["ep_reward"]
        .astype(float)
    )
    rows = []
    for attack in default_attacks:
        raw = (
            df[(df["attack_key"] == attack) & (df["stage"] == "attack")]
            .set_index("scenario_id")["ep_reward"]
            .astype(float)
        )
        defended = (
            df[(df["attack_key"] == attack) & (df["stage"] == "ug_bcr")]
            .set_index("scenario_id")["ep_reward"]
            .astype(float)
        )
        for scenario_id in sorted(set(clean.index) & set(raw.index) & set(defended.index)):
            attack_degradation = clean.loc[scenario_id] - raw.loc[scenario_id]
            defense_gain = defended.loc[scenario_id] - raw.loc[scenario_id]
            if attack_degradation <= 0:
                continue
            rows.append(
                {
                    "algorithm": algorithm.upper(),
                    "scenario_id": scenario_id,
                    "attack_key": attack,
                    "recovery": defense_gain / attack_degradation,
                    "clean_raw_reward": clean.loc[scenario_id],
                    "attack_raw_reward": raw.loc[scenario_id],
                    "attack_dtsr_reward": defended.loc[scenario_id],
                    "source": str(path),
                }
            )
    return pd.DataFrame(rows)


def _paired_remaining_from_csv(path: Path, algorithm_filter: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if algorithm_filter is not None:
        df = df[df["algorithm"].str.lower().isin(algorithm_filter)].copy()
    df = df[df["attack_key"].isin(["opposite_fgsm", "electhacker_c", "electhacker_f", "electhacker_o"])].copy()
    df["algorithm"] = df["algorithm"].str.upper()
    df["source"] = str(path)
    keep = [
        "algorithm",
        "scenario_id",
        "attack_key",
        "recovery",
        "clean_raw_reward",
        "attack_raw_reward",
        "attack_dtsr_reward",
        "source",
    ]
    return df[keep]


def build_samples(project_root: Path) -> pd.DataFrame:
    results = project_root / "results"
    frames = [
        _paired_default_from_online_vs(
            results
            / "online_vs_dtsr_default_attacks_seed42_newlong_v3_sealed"
            / "tables"
            / "online_vs_dtsr_episode_metrics_long.csv"
        ),
        _paired_default_from_stage_rollouts(
            "td3", results / "native_td3_dtsr_seed42" / "test_evaluation" / "rollouts.csv"
        ),
        _paired_default_from_stage_rollouts(
            "sac", results / "native_sac_dtsr_seed42" / "test_evaluation" / "rollouts.csv"
        ),
        _paired_default_from_stage_rollouts(
            "ppo",
            results
            / "native_ppo_dtsr_seed42"
            / "test_evaluation_same20_frozen"
            / "rollouts.csv",
        ),
        _paired_remaining_from_csv(
            results
            / "remaining4_cross_backbone_dtsr_seed42"
            / "paired_recovery_all_algorithms.csv",
            {"ddpg", "td3", "sac"},
        ),
        _paired_remaining_from_csv(
            results
            / "native_ppo_dtsr_seed42"
            / "remaining4_test_evaluation_same20_frozen"
            / "paired_recovery.csv",
            {"ppo"},
        ),
    ]
    samples = pd.concat(frames, ignore_index=True)
    samples = samples[samples["algorithm"].isin(ALGORITHMS)].copy()
    samples = samples[samples["attack_key"].isin(ATTACK_ORDER)].copy()
    samples["attack_label"] = samples["attack_key"].map(ATTACK_LABELS)
    samples["recovery_pct"] = samples["recovery"].astype(float) * 100.0
    samples["algorithm"] = pd.Categorical(samples["algorithm"], categories=ALGORITHMS, ordered=True)
    samples["attack_key"] = pd.Categorical(samples["attack_key"], categories=ATTACK_ORDER, ordered=True)
    return samples.sort_values(["algorithm", "attack_key", "scenario_id"]).reset_index(drop=True)


def write_summary(samples: pd.DataFrame, output_path: Path) -> None:
    summary = (
        samples.groupby(["algorithm", "attack_key"], observed=False)
        .agg(
            n=("recovery_pct", "count"),
            mean_pct=("recovery_pct", "mean"),
            std_pct=("recovery_pct", "std"),
            median_pct=("recovery_pct", "median"),
            q1_pct=("recovery_pct", lambda s: s.quantile(0.25)),
            q3_pct=("recovery_pct", lambda s: s.quantile(0.75)),
            min_pct=("recovery_pct", "min"),
            max_pct=("recovery_pct", "max"),
        )
        .reset_index()
    )
    summary["attack_label"] = summary["attack_key"].map(ATTACK_LABELS)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")


def style_axis(ax) -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.76)
    ax.tick_params(axis="x", length=2.4, width=0.68)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.42, zorder=0)
    ax.grid(axis="y", visible=False)


def draw_boxplot(
    samples: pd.DataFrame,
    output_stem: Path,
    *,
    hide_dots: bool,
    max_dots_per_box: int,
    dot_size: float,
    dot_alpha: float,
) -> None:
    configure_matplotlib()
    rng = np.random.default_rng(42)
    spacing = 0.48
    y_positions = {
        attack: (len(ATTACK_ORDER) - 1 - i) * spacing
        for i, attack in enumerate(ATTACK_ORDER)
    }
    yticks = [y_positions[a] for a in ATTACK_ORDER]
    ylabels = [ATTACK_LABELS[a] for a in ATTACK_ORDER]
    ymin = -0.24
    ymax = (len(ATTACK_ORDER) - 1) * spacing + 0.24

    fig, axes = plt.subplots(1, len(ALGORITHMS), figsize=(7.16, 2.05), facecolor="white")
    fig.subplots_adjust(left=0.14, right=0.985, top=0.86, bottom=0.26, wspace=0.10)

    for col, algorithm in enumerate(ALGORITHMS):
        ax = axes[col]
        sub = samples[samples["algorithm"].astype(str) == algorithm]
        for attack in ATTACK_ORDER:
            values = sub[sub["attack_key"].astype(str) == attack]["recovery_pct"].to_numpy(float)
            if len(values) == 0:
                continue
            color = ATTACK_COLORS[attack]
            bp = ax.boxplot(
                [values],
                positions=[y_positions[attack]],
                vert=False,
                widths=0.24,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                whis=(5, 95),
                medianprops={"color": "#333333", "linewidth": 0.75},
                boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.55, "alpha": 0.70},
                whiskerprops={"color": color, "linewidth": 0.65},
                capprops={"color": color, "linewidth": 0.65},
            )
            for patch in bp["boxes"]:
                patch.set_zorder(2)
            if not hide_dots and max_dots_per_box > 0:
                if len(values) > max_dots_per_box:
                    display_values = rng.choice(values, size=max_dots_per_box, replace=False)
                else:
                    display_values = values
                jitter = rng.uniform(-0.065, 0.065, size=len(display_values))
                ax.scatter(
                    display_values,
                    y_positions[attack] + jitter,
                    s=dot_size,
                    color="#1F1F1F",
                    alpha=dot_alpha,
                    linewidths=0,
                    zorder=3,
                )

        ax.set_xlim(55, 105)
        ax.set_xticks([60, 70, 80, 90, 100])
        ax.set_ylim(ymin, ymax)
        ax.set_title(algorithm, pad=3.0, fontweight="bold")
        ax.set_yticks(yticks)
        if col == 0:
            ax.set_yticklabels(ylabels, fontsize=7.0)
            ax.tick_params(axis="y", length=0, pad=3.8)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        style_axis(ax)
        ax.tick_params(axis="x", pad=1.5)

    fig.supxlabel("Attack recovery rate (%)", x=0.5625, y=0.055, fontsize=8.0)
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "cross_backbone_recovery_figure",
    )
    parser.add_argument("--output-name", default="figure_cross_backbone_boxplot")
    parser.add_argument("--hide-dots", action="store_true")
    parser.add_argument("--max-dots-per-box", type=int, default=20)
    parser.add_argument("--dot-size", type=float, default=3.0)
    parser.add_argument("--dot-alpha", type=float, default=0.20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(args.project_root)
    samples_path = args.output_dir / "cross_backbone_boxplot_samples.csv"
    summary_path = args.output_dir / "figure_cross_backbone_boxplot_summary.csv"
    output_stem = args.output_dir / args.output_name
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    write_summary(samples, summary_path)
    draw_boxplot(
        samples,
        output_stem,
        hide_dots=args.hide_dots,
        max_dots_per_box=args.max_dots_per_box,
        dot_size=args.dot_size,
        dot_alpha=args.dot_alpha,
    )
    print(output_stem.with_suffix(".pdf"))
    print(samples_path)
    print(summary_path)


if __name__ == "__main__":
    main()
