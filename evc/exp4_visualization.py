from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KNOWLEDGE_ORDER = ["K0", "K1", "K2", "K3", "K4"]
KNOWLEDGE_LABELS = {
    "K0": "K0\nActor",
    "K1": "K1\n+DAE",
    "K2": "K2\n+DET",
    "K3": "K3\n+UG-BCR",
    "K4": "K4\nFull DTSR",
}
COLORS = {
    "attack": "#D55E00",
    "full": "#00796B",
    "recovery": "#3366A8",
    "harm": "#B22222",
    "clean": "#4D4D4D",
    "scenario": "#A7ADB4",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D9DDE1",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rollouts(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise RuntimeError(f"No rollout rows found in {path}")
    return pd.DataFrame(rows)


def _complete_restarts(raw_df: pd.DataFrame) -> pd.DataFrame:
    attacks = raw_df[raw_df["condition_key"] != "clean"].copy()
    if attacks.empty:
        return attacks
    stage_counts = attacks.groupby(
        ["scenario_id", "condition_key", "restart"], sort=False
    )["stage_key"].nunique()
    complete = stage_counts[stage_counts >= 2].reset_index()[
        ["scenario_id", "condition_key", "restart"]
    ]
    return attacks.merge(
        complete, on=["scenario_id", "condition_key", "restart"], how="inner"
    )


def select_restart_per_condition(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Select only among restarts of the same condition; never across K levels."""
    attacks = _complete_restarts(raw_df)
    if attacks.empty:
        return attacks
    clean = raw_df[raw_df["condition_key"] == "clean"].copy()
    clean_full = clean[clean["stage_key"] == "full_dtsr"].set_index("scenario_id")
    selected: list[pd.DataFrame] = []
    for (scenario_id, condition_key), group in attacks.groupby(
        ["scenario_id", "condition_key"], sort=False
    ):
        candidates: list[dict[str, float]] = []
        for restart, restart_group in group.groupby("restart", sort=False):
            by_stage = restart_group.set_index("stage_key")
            if not {"attack", "full_dtsr"}.issubset(by_stage.index):
                continue
            full = by_stage.loc["full_dtsr"]
            candidates.append(
                {
                    "restart": int(restart),
                    "full_reward": float(full["ep_reward"]),
                    "full_cost": float(full["ep_r1"]),
                    "full_exit": float(full["exit_vio"]),
                }
            )
        rank = pd.DataFrame(candidates)
        if rank.empty:
            continue
        objective = str(group["objective"].iloc[0])
        if objective == "economic":
            clean_exit = (
                float(clean_full.loc[scenario_id, "exit_vio"])
                if scenario_id in clean_full.index
                else 0.0
            )
            feasible = rank[rank["full_exit"] <= clean_exit].copy()
            if feasible.empty:
                feasible = rank.sort_values(
                    ["full_exit", "full_cost"], ascending=[True, False]
                )
            else:
                feasible = feasible.sort_values(
                    ["full_cost", "full_reward"], ascending=[False, True]
                )
            chosen = int(feasible.iloc[0]["restart"])
        else:
            chosen = int(
                rank.sort_values(
                    ["full_exit", "full_reward"], ascending=[False, True]
                ).iloc[0]["restart"]
            )
        selected.append(group[group["restart"].astype(int) == chosen])
    return pd.concat(selected, ignore_index=True) if selected else attacks.iloc[0:0]


def build_paired_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    selected = select_restart_per_condition(raw_df)
    clean = raw_df[raw_df["condition_key"] == "clean"].copy()
    clean_attack = clean[clean["stage_key"] == "attack"][
        ["scenario_id", "ep_reward", "ep_r1", "exit_vio", "done_cnt", "mean_fin_soc"]
    ].rename(
        columns={
            "ep_reward": "clean_attack_reward",
            "ep_r1": "clean_attack_cost",
            "exit_vio": "clean_attack_exit_vio",
            "done_cnt": "clean_attack_done_cnt",
            "mean_fin_soc": "clean_attack_fin_soc",
        }
    )
    clean_full = clean[clean["stage_key"] == "full_dtsr"][
        ["scenario_id", "ep_reward", "ep_r1", "exit_vio", "done_cnt", "mean_fin_soc"]
    ].rename(
        columns={
            "ep_reward": "clean_full_reward",
            "ep_r1": "clean_full_cost",
            "exit_vio": "clean_full_exit_vio",
            "done_cnt": "clean_full_done_cnt",
            "mean_fin_soc": "clean_full_fin_soc",
        }
    )
    baseline = clean_attack.merge(clean_full, on="scenario_id", how="inner")
    rows: list[pd.DataFrame] = []
    for condition_key, group in selected.groupby("condition_key", sort=False):
        attack = group[group["stage_key"] == "attack"][
            [
                "scenario_id",
                "episode_index",
                "target",
                "objective",
                "knowledge",
                "ep_reward",
                "ep_r1",
                "exit_vio",
                "done_cnt",
                "mean_fin_soc",
            ]
        ].rename(
            columns={
                "ep_reward": "attack_reward",
                "ep_r1": "attack_cost",
                "exit_vio": "attack_exit_vio",
                "done_cnt": "attack_done_cnt",
                "mean_fin_soc": "attack_fin_soc",
            }
        )
        full_columns = [
            "scenario_id",
            "ep_reward",
            "ep_r1",
            "exit_vio",
            "done_cnt",
            "mean_fin_soc",
            "route_rate",
            "urgency_gate_belief_rate",
            "shield_correction_mean",
        ]
        full = group[group["stage_key"] == "full_dtsr"][full_columns].rename(
            columns={
                "ep_reward": "full_reward",
                "ep_r1": "full_cost",
                "exit_vio": "full_exit_vio",
                "done_cnt": "full_done_cnt",
                "mean_fin_soc": "full_fin_soc",
            }
        )
        merged = attack.merge(full, on="scenario_id", how="inner").merge(
            baseline, on="scenario_id", how="inner"
        )
        if merged.empty:
            continue
        denominator = merged["clean_attack_reward"] - merged["attack_reward"]
        merged["recovery_pct"] = np.where(
            denominator.abs() > 1e-8,
            100.0 * (merged["full_reward"] - merged["attack_reward"]) / denominator,
            np.nan,
        )
        merged["attack_exit_rate_pct"] = (
            100.0 * merged["attack_exit_vio"] / merged["attack_done_cnt"].clip(lower=1.0)
        )
        merged["full_exit_rate_pct"] = (
            100.0 * merged["full_exit_vio"] / merged["full_done_cnt"].clip(lower=1.0)
        )
        merged["defense_gain"] = merged["full_reward"] - merged["attack_reward"]
        merged["full_cost_increase"] = merged["full_cost"] - merged["clean_full_cost"]
        merged["full_cost_increase_pct"] = (
            100.0 * merged["full_cost_increase"] / merged["clean_full_cost"].abs().clip(lower=1e-8)
        )
        merged["attack_cost_increase_pct"] = (
            100.0
            * (merged["attack_cost"] - merged["clean_attack_cost"])
            / merged["clean_attack_cost"].abs().clip(lower=1e-8)
        )
        merged["condition_key"] = condition_key
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def _common_deadline_scenarios(paired: pd.DataFrame) -> pd.DataFrame:
    deadline = paired[paired["objective"] == "deadline"].copy()
    coverage = deadline.groupby("scenario_id")["knowledge"].nunique()
    complete_ids = coverage[coverage >= len(KNOWLEDGE_ORDER)].index
    deadline = deadline[deadline["scenario_id"].isin(complete_ids)].copy()
    deadline["knowledge"] = pd.Categorical(
        deadline["knowledge"], categories=KNOWLEDGE_ORDER, ordered=True
    )
    return deadline.sort_values(["episode_index", "knowledge"])


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_knowledge_ladder(deadline: pd.DataFrame, output_dir: Path, seed: int) -> None:
    scenario_count = deadline["scenario_id"].nunique()
    if scenario_count == 0:
        return
    x = np.arange(len(KNOWLEDGE_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    specs = [
        ("recovery_pct", "Recovery (%)", COLORS["recovery"]),
        ("full_exit_rate_pct", "Residual exit violation (%)", COLORS["harm"]),
    ]
    for panel_index, (metric, ylabel, color) in enumerate(specs):
        ax = axes[panel_index]
        pivot = deadline.pivot(index="scenario_id", columns="knowledge", values=metric).reindex(
            columns=KNOWLEDGE_ORDER
        )
        for values in pivot.to_numpy(dtype=float):
            ax.plot(x, values, color=COLORS["scenario"], linewidth=0.75, alpha=0.42, zorder=1)
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for index, knowledge in enumerate(KNOWLEDGE_ORDER):
            mean, low, high = _bootstrap_ci(
                deadline.loc[deadline["knowledge"] == knowledge, metric].to_numpy(dtype=float),
                seed + panel_index * 100 + index,
            )
            means.append(mean)
            lows.append(low)
            highs.append(high)
        means_array = np.asarray(means)
        ax.fill_between(x, lows, highs, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, means_array, color=color, linewidth=2.2, marker="o", markersize=5.0, zorder=3)
        for index, value in enumerate(means_array):
            ax.annotate(
                f"{value:.1f}",
                (x[index], value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=8,
                fontweight="bold",
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [KNOWLEDGE_LABELS[key] for key in KNOWLEDGE_ORDER])
        ax.grid(axis="y")
        ax.margins(x=0.06)
    fig.suptitle(
        f"Adaptive knowledge ladder under Deadline Denial (n={scenario_count} paired scenarios)",
        x=0.06,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93), w_pad=2.2)
    for panel_index, ax in enumerate(axes):
        position = ax.get_position()
        fig.text(
            0.5 * (position.x0 + position.x1),
            0.025,
            f"({chr(97 + panel_index)}) {specs[panel_index][1]}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    _save_figure(fig, output_dir / "fig_exp4_knowledge_ladder")


def plot_attack_defense_gap(deadline: pd.DataFrame, output_dir: Path, seed: int) -> None:
    scenario_count = deadline["scenario_id"].nunique()
    if scenario_count == 0:
        return
    fig, ax = plt.subplots(figsize=(6.7, 3.6))
    y = np.arange(len(KNOWLEDGE_ORDER))[::-1]
    clean_mean = float(deadline.groupby("scenario_id")["clean_attack_reward"].first().mean())
    ax.axvline(clean_mean, color=COLORS["clean"], linestyle="--", linewidth=1.1, alpha=0.8, label="Clean return")
    for row_index, knowledge in enumerate(KNOWLEDGE_ORDER):
        group = deadline[deadline["knowledge"] == knowledge]
        attack_mean, attack_low, attack_high = _bootstrap_ci(
            group["attack_reward"].to_numpy(dtype=float), seed + row_index
        )
        full_mean, full_low, full_high = _bootstrap_ci(
            group["full_reward"].to_numpy(dtype=float), seed + 50 + row_index
        )
        ypos = y[row_index]
        ax.plot([attack_mean, full_mean], [ypos, ypos], color="#9CA3AA", linewidth=2.0, zorder=1)
        ax.errorbar(
            attack_mean,
            ypos,
            xerr=[[attack_mean - attack_low], [attack_high - attack_mean]],
            fmt="o",
            color=COLORS["attack"],
            capsize=2.5,
            markersize=5.5,
            zorder=3,
            label="Attack-only" if row_index == 0 else None,
        )
        ax.errorbar(
            full_mean,
            ypos,
            xerr=[[full_mean - full_low], [full_high - full_mean]],
            fmt="s",
            color=COLORS["full"],
            capsize=2.5,
            markersize=5.2,
            zorder=3,
            label="Full DTSR" if row_index == 0 else None,
        )
        ax.annotate(
            f"+{full_mean - attack_mean:.0f}",
            ((attack_mean + full_mean) / 2.0, ypos),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color="#555555",
            fontsize=7.5,
        )
    ax.set_yticks(y, [KNOWLEDGE_LABELS[key].replace("\n", "  ") for key in KNOWLEDGE_ORDER])
    ax.set_xlabel("Episode return (higher is better)")
    ax.set_title(
        f"Defense recovery gap for each knowledge level (n={scenario_count})",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.57, 0.91))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save_figure(fig, output_dir / "fig_exp4_attack_defense_gap")


def write_figure_manifest(output_dir: Path, deadline: pd.DataFrame) -> None:
    manifest = {
        "knowledge_summary_mode": "single_knowledge_level",
        "cumulative_best_of_across_knowledge": False,
        "uncertainty": "scenario-level nonparametric bootstrap 95% CI, 5000 resamples",
        "deadline_common_scenarios": int(deadline["scenario_id"].nunique()),
        "figures": [
            "fig_exp4_knowledge_ladder",
            "fig_exp4_attack_defense_gap",
        ],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    captions = """# Experiment 4 figure captions

## Knowledge ladder
Mean recovery and residual departure-SOC violation as the attacker receives progressively more module knowledge. Thin gray lines are paired test scenarios; the colored line is the single-K mean and shading is the scenario-level bootstrap 95% confidence interval. No best-of aggregation is performed across K0-K4.

## Attack-defense gap
Mean episode return before and after the full DTSR pipeline for each independently optimized knowledge level. Horizontal intervals are bootstrap 95% confidence intervals and the connecting segment shows reward recovered by DTSR.
"""
    (output_dir / "figure_captions.md").write_text(captions, encoding="utf-8")


def create_exp4_figures(raw_path: Path, output_dir: Path, seed: int = 42) -> dict[str, int]:
    _configure_style()
    raw_df = load_rollouts(raw_path)
    paired = build_paired_metrics(raw_df)
    if paired.empty:
        raise RuntimeError("No complete attack/full-DTSR condition pairs are available for plotting.")
    deadline = _common_deadline_scenarios(paired)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_knowledge_ladder(deadline, output_dir, seed)
    plot_attack_defense_gap(deadline, output_dir, seed)
    write_figure_manifest(output_dir, deadline)
    return {
        "raw_rows": int(len(raw_df)),
        "paired_rows": int(len(paired)),
        "deadline_common_scenarios": int(deadline["scenario_id"].nunique()),
    }
