"""
visualize/ablation_plot.py
===========================
Generate bar and line plots for the ablation studies.

Supports both forecasting (regression) and classification results.
Works on stock and index datasets — just point ``--results-dir`` at the
appropriate folder.

Run
---
    python -m visualize.ablation_plot --results-dir results
    python -m visualize.ablation_plot --results-dir results/index_results --task classification
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import ablation helpers
from evaluation.forecasting_ablation import (
    load_and_decode as load_regression,
    _best_model_in_group,
    BASELINE_CATEGORIES,
)
from evaluation.classification_ablation import (
    load_and_decode as load_classification,
    CLF_BASELINE_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Plotting style constants
# ---------------------------------------------------------------------------

_DISTANCE_ORDER = ["Squared Correlation", "Partial Correlation", "Mutual Information"]
_STRUCTURE_ORDER = ["Plain", "Clustering", "Split", "Split+Clustering"]
_WEIGHTING_ORDER = ["IDW", "Exp", "Learned"]

_DISTANCE_COLORS = {
    "Squared Correlation": "#1f77b4",
    "Partial Correlation": "#ff7f0e",
    "Mutual Information": "#2ca02c",
}
_STRUCTURE_COLORS = {
    "Plain": "#1f77b4",
    "Clustering": "#ff7f0e",
    "Split": "#2ca02c",
    "Split+Clustering": "#d62728",
}
_WEIGHTING_COLORS = {
    "IDW": "#1f77b4",
    "Exp": "#ff7f0e",
    "Learned": "#2ca02c",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Bar / line annotation helper
# ---------------------------------------------------------------------------

def _smart_round(v: float) -> str:
    """Round a value to a sensible number of decimal places for display."""
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    if av >= 0.01:
        return f"{v:.3f}"
    return f"{v:.4f}"


def _annotate_bars(ax: plt.Axes, fontsize: int = 7):
    """Place rounded value labels above (or below) every bar patch."""
    for p in ax.patches:
        h = p.get_height()
        if not np.isfinite(h):
            continue
        ax.annotate(
            _smart_round(h),
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=fontsize,
            textcoords="offset points",
            xytext=(0, 3 if h >= 0 else -3),
        )


def _annotate_line_points(ax: plt.Axes, fontsize: int = 7):
    """Place rounded value labels next to every data point on line plots."""
    for line in ax.lines:
        xd, yd = line.get_xdata(), line.get_ydata()
        for x, y in zip(xd, yd):
            if np.isfinite(y):
                ax.annotate(
                    _smart_round(y),
                    (x, y),
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    textcoords="offset points",
                    xytext=(0, 4),
                )


# ---------------------------------------------------------------------------
# Y-axis zoom helper: tighten axis limits around the data range so that small
# differences between near-equal bars / lines are visible.
# ---------------------------------------------------------------------------

def _zoom_y(ax: plt.Axes, lower_is_better: bool, margin_frac: float = 0.3):
    """Adjust y-limits to show ±margin around the data range."""
    lo, hi = ax.get_ylim()
    data_lo, data_hi = hi, lo  # will scan for actual data range
    for coll in list(ax.patches) + list(ax.lines):
        try:
            ys = [p.get_height() + p.get_y() for p in ax.patches]
            break
        except Exception:
            pass
    else:
        ys = []
    for line in ax.lines:
        yd = line.get_ydata()
        if len(yd):
            ys.extend(yd)
    ys = [y for y in ys if np.isfinite(y)]
    if not ys:
        return
    data_lo, data_hi = min(ys), max(ys)
    span = data_hi - data_lo
    if span < 1e-12:
        span = abs(data_hi) * 0.01 or 0.01
    margin = span * margin_frac
    ax.set_ylim(data_lo - margin, data_hi + margin)


# ---------------------------------------------------------------------------
# Helper: aggregate best metric per dimension value
# ---------------------------------------------------------------------------

def _agg_best_per_group(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    lower_is_better: bool,
    filter_col: str | None = None,
    filter_val: str | None = None,
) -> pd.DataFrame:
    """For each *group_col* value, find the best (Category, Model) by mean
    *metric* across (Ticker, Fold), return a summary row."""
    sub = df[df[group_col].notna()].copy()
    if filter_col and filter_val is not None:
        sub = sub[sub[filter_col] == filter_val]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for val, g in sub.groupby(group_col):
        best = _best_model_in_group(g, metric, lower_is_better)
        rows.append({group_col: val, f"mean_{metric}": best[metric]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Best k line/bar chart (one series per distance metric)
# ---------------------------------------------------------------------------

def plot_best_k(
    df: pd.DataFrame,
    metric: str,
    lower_is_better: bool,
    save_path: Path | None = None,
    title_suffix: str = "",
):
    """Line plot: x = k, y = best mean metric, one line per distance."""
    net = df[df["k"].notna()].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for dist in _DISTANCE_ORDER:
        g_dist = net[net["distance"] == dist]
        if g_dist.empty:
            continue
        k_vals = sorted(g_dist["k"].unique())
        means = []
        for k in k_vals:
            g_k = g_dist[g_dist["k"] == k]
            best = _best_model_in_group(g_k, metric, lower_is_better)
            means.append(best[metric])
        ax.plot([int(k) for k in k_vals], means, "o-",
                label=dist, color=_DISTANCE_COLORS.get(dist))

    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel(f"Best mean {metric}")
    ax.set_title(f"Effect of k on Best Model Performance{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _annotate_line_points(ax)
    if save_path:
        fig.savefig(save_path)
        print(f"  → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Feature structure bar chart (grouped by distance)
# ---------------------------------------------------------------------------

def plot_structure(
    df: pd.DataFrame,
    metric: str,
    lower_is_better: bool,
    save_path: Path | None = None,
    title_suffix: str = "",
):
    """Grouped bar chart: groups = distance, bars = feature structure."""
    net = df[df["structure"].notna()].copy()
    distances = [d for d in _DISTANCE_ORDER if d in net["distance"].unique()]
    structures = [s for s in _STRUCTURE_ORDER if s in net["structure"].unique()]

    if not distances or not structures:
        return

    x = np.arange(len(distances))
    width = 0.8 / len(structures)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, struct in enumerate(structures):
        vals = []
        for dist in distances:
            sub = net[(net["distance"] == dist) & (net["structure"] == struct)]
            if sub.empty:
                vals.append(np.nan)
            else:
                best = _best_model_in_group(sub, metric, lower_is_better)
                vals.append(best[metric])
        offset = (i - len(structures) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=struct,
               color=_STRUCTURE_COLORS.get(struct))

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" ", "\n") for d in distances])
    ax.set_ylabel(f"Best mean {metric}")
    ax.set_title(f"Feature Structure Comparison{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _annotate_bars(ax)
    _zoom_y(ax, lower_is_better)
    if save_path:
        fig.savefig(save_path)
        print(f"  → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Distance metric bar chart (grouped by structure)
# ---------------------------------------------------------------------------

def plot_distance(
    df: pd.DataFrame,
    metric: str,
    lower_is_better: bool,
    save_path: Path | None = None,
    title_suffix: str = "",
):
    """Grouped bar chart: groups = structure, bars = distance."""
    net = df[df["distance"].notna()].copy()
    structures = [s for s in _STRUCTURE_ORDER if s in net["structure"].unique()]
    distances = [d for d in _DISTANCE_ORDER if d in net["distance"].unique()]

    if not structures or not distances:
        return

    x = np.arange(len(structures))
    width = 0.8 / len(distances)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, dist in enumerate(distances):
        vals = []
        for struct in structures:
            sub = net[(net["structure"] == struct) & (net["distance"] == dist)]
            if sub.empty:
                vals.append(np.nan)
            else:
                best = _best_model_in_group(sub, metric, lower_is_better)
                vals.append(best[metric])
        offset = (i - len(distances) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=dist,
               color=_DISTANCE_COLORS.get(dist))

    ax.set_xticks(x)
    ax.set_xticklabels(structures)
    ax.set_ylabel(f"Best mean {metric}")
    ax.set_title(f"Distance Metric Comparison{title_suffix}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _annotate_bars(ax)
    _zoom_y(ax, lower_is_better)
    if save_path:
        fig.savefig(save_path)
        print(f"  → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Weighting scheme bar chart
# ---------------------------------------------------------------------------

def plot_weighting(
    df: pd.DataFrame,
    metric: str,
    lower_is_better: bool,
    save_path: Path | None = None,
    title_suffix: str = "",
):
    """Simple bar chart: weighting scheme vs best metric."""
    net = df[df["weighting"].notna()].copy()
    schemes = [w for w in _WEIGHTING_ORDER if w in net["weighting"].unique()]
    if not schemes:
        return

    vals = []
    for w in schemes:
        g = net[net["weighting"] == w]
        best = _best_model_in_group(g, metric, lower_is_better)
        vals.append(best[metric])

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [_WEIGHTING_COLORS.get(w, "#999") for w in schemes]
    ax.bar(schemes, vals, color=colors)
    ax.set_ylabel(f"Best mean {metric}")
    ax.set_title(f"Weighting Scheme Comparison{title_suffix}")
    ax.grid(True, alpha=0.3, axis="y")
    _annotate_bars(ax)
    _zoom_y(ax, lower_is_better)
    if save_path:
        fig.savefig(save_path)
        print(f"  → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Baseline vs overall-best-network bar
# ---------------------------------------------------------------------------

def plot_baseline_vs_network(
    df: pd.DataFrame,
    metric: str,
    lower_is_better: bool,
    baseline_cats: frozenset,
    save_path: Path | None = None,
    title_suffix: str = "",
):
    baselines = df[df["Category"].isin(baseline_cats)]
    networks = df[df["k"].notna()]
    if baselines.empty or networks.empty:
        return

    bl_best = _best_model_in_group(baselines, metric, lower_is_better)
    net_best = _best_model_in_group(networks, metric, lower_is_better)

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [
        f"Baseline\n({bl_best['Category']})",
        f"Network\n({net_best['Category']})",
    ]
    vals = [bl_best[metric], net_best[metric]]
    colors = ["#ff7f0e", "#1f77b4"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel(f"Best mean {metric}")
    ax.set_title(f"Baseline vs Best Network{title_suffix}")
    ax.grid(True, alpha=0.3, axis="y")
    _annotate_bars(ax)
    _zoom_y(ax, lower_is_better)
    if save_path:
        fig.savefig(save_path)
        print(f"  → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master driver
# ---------------------------------------------------------------------------

def run_ablation_plots(
    df: pd.DataFrame,
    metric: str,
    lower_is_better: bool,
    baseline_cats: frozenset,
    save_dir: Path,
    task_label: str = "regression",
    dataset_label: str = "stock",
):
    save_dir.mkdir(parents=True, exist_ok=True)
    for stale in save_dir.glob(f"{task_label}_ablation_*.png"):
        stale.unlink()
    suffix = f" ({dataset_label}, {task_label}, {metric})"

    print(f"\nGenerating ablation plots for {task_label} ({metric}) ...")

    plot_best_k(df, metric, lower_is_better,
                save_dir / f"{task_label}_ablation_k.png", suffix)
    plot_structure(df, metric, lower_is_better,
                   save_dir / f"{task_label}_ablation_structure.png", suffix)
    plot_distance(df, metric, lower_is_better,
                  save_dir / f"{task_label}_ablation_distance.png", suffix)
    plot_weighting(df, metric, lower_is_better,
                   save_dir / f"{task_label}_ablation_weighting.png", suffix)
    plot_baseline_vs_network(df, metric, lower_is_better, baseline_cats,
                             save_dir / f"{task_label}_ablation_baseline_vs_net.png",
                             suffix)

    print(f"  All plots saved to {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_results_dirs() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    return [repo_root / "results", repo_root / "results" / "index_results"]


def _label_for_results_dir(results_dir: Path) -> str:
    return "index" if results_dir.name == "index_results" else "stock"

def main():
    parser = argparse.ArgumentParser(description="Ablation study plots")
    parser.add_argument("--results-dir", action="append", default=None,
                        help="Directory containing results JSON files. Repeat to process multiple folders. Defaults to stock + index.")
    parser.add_argument("--task", type=str, default="both",
                        choices=["regression", "classification", "both"],
                        help="Which task to plot (default: both)")
    parser.add_argument("--metric", type=str, default=None,
                        help="Override metric (default: RMSE_log / ROC_AUC)")
    args = parser.parse_args()

    results_dirs = [Path(p) for p in args.results_dir] if args.results_dir else _default_results_dirs()

    for results_dir in results_dirs:
        dataset_label = _label_for_results_dir(results_dir)
        save_dir = results_dir / "plots" / "ablation"

        if args.task in ("regression", "both"):
            reg_path = results_dir / "results_bench.json"
            if reg_path.exists():
                metric = args.metric or "RMSE_log"
                lower = metric not in ("R2", "R2_log")
                df_reg = load_regression(reg_path)
                print(f"Regression [{dataset_label}]: {len(df_reg)} rows, "
                      f"{df_reg['Ticker'].nunique()} tickers")
                run_ablation_plots(df_reg, metric, lower, BASELINE_CATEGORIES,
                                 save_dir, "regression", dataset_label)
            else:
                print(f"Skipping regression [{dataset_label}] — {reg_path} not found")

        if args.task in ("classification", "both"):
            clf_path = results_dir / "classification_results.json"
            if clf_path.exists():
                metric = args.metric or "ROC_AUC"
                lower = False
                df_clf = load_classification(clf_path)
                print(f"Classification [{dataset_label}]: {len(df_clf)} rows, "
                      f"{df_clf['Ticker'].nunique()} tickers")
                run_ablation_plots(df_clf, metric, lower, CLF_BASELINE_CATEGORIES,
                                 save_dir, "classification", dataset_label)
            else:
                print(f"Skipping classification [{dataset_label}] — {clf_path} not found")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
