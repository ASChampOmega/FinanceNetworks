import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Category classification helpers
# ---------------------------------------------------------------------------

_BASELINE_CATEGORIES = frozenset({"HAR", "ARIMA", "GARCH", "RegimeSwitching"})
_NON_GARCH_BASELINES = frozenset({"HAR", "ARIMA", "RegimeSwitching"})


def _strip_k(cat: str) -> str:
    """Remove [k=N] suffix from a category string."""
    return re.sub(r"\s*\[k=\d+\]$", "", cat)


def _distance_group(cat: str) -> str:
    """Map a category string to its distance-metric group."""
    base = _strip_k(cat)
    if base.startswith("PCorr"):
        return "PCorr"
    if base.startswith("MI ") or base == "MI":
        return "MI"
    return "Corr"


def _is_network(cat: str) -> bool:
    return _strip_k(cat) not in _BASELINE_CATEGORIES


# ---------------------------------------------------------------------------
# plot_ticker_predictions  (rewritten)
# ---------------------------------------------------------------------------

def plot_ticker_predictions(
    pred_store: dict,
    metrics_df: pd.DataFrame,
    sample_tickers: list,
    save_dir: "str | None" = None,
    use_log: bool = True,
):
    """
    For each ticker plot true Y_fwd against a small set of selected models:

    * **Best Corr network** — lowest per-ticker RMSE among all squared-
      correlation network categories (Network, Clustering, SplitFeatures,
      ExpKernel, LearnedWeight, …).
    * **Best PCorr network** — same, among PCorr* categories.
    * **Best MI network** — same, among MI* categories.
    * **Best GARCH** — lowest per-ticker RMSE among GARCH models.
    * **Best baseline** — lowest per-ticker RMSE among HAR / ARIMA /
      RegimeSwitching (i.e. all non-GARCH baselines).

    Selection is done **per ticker** so each stock gets its own best model
    from each group.

    Parameters
    ----------
    pred_store     : ``{ticker: DataFrame}`` with ``Y_true`` + model columns.
    metrics_df     : Fold-level metrics (has Category, Ticker, Model, RMSE, …).
    sample_tickers : Tickers to plot.
    save_dir       : If given, saves ``<save_dir>/<ticker>_predictions.png``.
    use_log        : Use RMSE_log (True) or raw RMSE (False) for selection.
    """
    # choose metric column based on selection (log vs raw)
    if "RMSE" not in metrics_df.columns and "RMSE_log" not in metrics_df.columns:
        raise ValueError("metrics_df must contain 'RMSE' or 'RMSE_log' column")
    rmse_col = "RMSE_log" if use_log and "RMSE_log" in metrics_df.columns else "RMSE"

    # Average RMSE per (Category, Ticker, Model)
    per_ticker = (
        metrics_df
        .groupby(["Category", "Ticker", "Model"])[rmse_col]
        .mean()
        .reset_index()
    )

    # Assign each category to a plotting group
    per_ticker["_group"] = per_ticker["Category"].apply(
        lambda c: (
            "GARCH" if _strip_k(c) == "GARCH"
            else ("Baseline" if _strip_k(c) in _NON_GARCH_BASELINES
                  else _distance_group(c))
        )
    )

    # Fixed colour scheme
    group_style = {
        "Corr":     {"color": "#1f77b4", "label": "Best Corr Network"},
        "PCorr":    {"color": "#ff7f0e", "label": "Best PCorr Network"},
        "MI":       {"color": "#2ca02c", "label": "Best MI Network"},
        "GARCH":    {"color": "#d62728", "label": "Best GARCH"},
        "Baseline": {"color": "#9467bd", "label": "Best Baseline (non-GARCH)"},
    }

    for ticker in sample_tickers:
        if ticker not in pred_store:
            continue

        df = pred_store[ticker]
        if "Y_true" not in df.columns:
            continue

        ticker_metrics = per_ticker[per_ticker["Ticker"] == ticker]
        if ticker_metrics.empty:
            continue

        # Pick best model per group for this ticker (lowest RMSE)
        curves = []   # (group_key, col_key, label)
        for grp_key, style in group_style.items():
            grp = ticker_metrics[ticker_metrics["_group"] == grp_key]
            if grp.empty:
                continue
            best_idx = grp[rmse_col].idxmin()
            best_cat   = grp.loc[best_idx, "Category"]
            best_model = grp.loc[best_idx, "Model"]
            best_rmse  = grp.loc[best_idx, rmse_col]
            col_key = f"[{best_cat}] {best_model}"
            if col_key in df.columns:
                display_model = best_model.replace(" (no outliers)", "")
                sel_tag = "log" if use_log else "raw"
                label = f"{style['label']} ({sel_tag}): {display_model}  ({rmse_col}={best_rmse:.4f})"
                curves.append((grp_key, col_key, label))

        if not curves:
            continue

        # Restrict to 2025 test period
        df = df[df.index.year == 2025]
        if df.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(
            df.index, df["Y_true"],
            label="True Y_fwd",
            color="black",
            linewidth=1.4,
            alpha=0.9,
            zorder=5,
        )

        for grp_key, col_key, label in curves:
            style = group_style[grp_key]
            ax.plot(
                df.index,
                df[col_key],
                label=label,
                linewidth=1.0,
                alpha=0.80,
                color=style["color"],
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_title(f"{ticker} — Model Predictions", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Realized Variance (Y_fwd)")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.7)
        fig.tight_layout()

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(f"{save_dir}/{ticker}_predictions.png", dpi=150, bbox_inches="tight")

        plt.show()
        plt.close(fig)


def plot_summary_metrics(
    summary_df: pd.DataFrame,
    save_path: "str | None" = None,
):
    """
    Side-by-side horizontal bar charts of mean_R2 and mean_RMSE for each model.
    Shows only the original-scale (non-log) metrics for interpretability.
    Models are coloured by category; a vertical dashed line marks zero on R2.
    """
    metrics = [
        ("mean_R2",   "Mean R\u00b2"),
        ("mean_RMSE", "Mean RMSE"),
    ]
    metrics = [(col, label) for col, label in metrics if col in summary_df.columns]

    n_models = len(summary_df)
    fig_h = max(5, n_models * 0.45)
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(7 * len(metrics), fig_h),
    )
    if len(metrics) == 1:
        axes = [axes]

    # Assign a colour per category
    if "Category" in summary_df.index.names:
        cats = summary_df.index.get_level_values("Category").unique().tolist()
    else:
        cats = []
    cat_colours = {}
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, c in enumerate(cats):
        cat_colours[c] = palette[i % len(palette)]

    for ax, (col, label) in zip(axes, metrics):
        data = summary_df[col].sort_values(ascending=True)
        bar_colours = []
        for idx in data.index:
            cat = idx[0] if isinstance(idx, tuple) else "Other"
            bar_colours.append(cat_colours.get(cat, "steelblue"))

        bars = ax.barh(
            [str(i) for i in data.index],
            data.values,
            color=bar_colours,
            edgecolor="white",
            height=0.65,
        )
        for bar, val in zip(bars, data.values):
            offset = max(abs(data.values)) * 0.01
            ax.text(
                val + offset if val >= 0 else val - offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
            )
        if col in ("mean_R2", "mean_R2_log"):
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(label)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Model Comparison Across Tickers & Folds",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_network_degrees(
    network,
    sample_tickers: "list[str] | None" = None,
    save_path: "str | None" = None,
):
    """
    Plot rolling graph degree over time for a selection of tickers.

    Shows how each stock's connectivity in the correlation network evolves,
    with the cross-sectional mean degree in the background.  Sharp drops or
    spikes in connectivity often coincide with volatility regime changes.

    Parameters
    ----------
    network       : a fitted FinanceNetworkBase subclass (after fit_transform).
    sample_tickers: tickers to highlight; defaults to up to 5 tickers.
    save_path     : if provided, saves the figure to this path.
    """
    deg_df = network.snapshot_degrees()

    if deg_df.empty:
        print("[plot_network_degrees] No snapshot data available.")
        return

    available = [t for t in (sample_tickers or []) if t in deg_df.columns]
    if not available:
        available = deg_df.columns[deg_df.notna().any()].tolist()[:5]

    fig, ax = plt.subplots(figsize=(14, 8))

    mean_deg = deg_df.mean(axis=1)
    ax.fill_between(
        deg_df.index,
        mean_deg,
        alpha=0.12,
        color="steelblue",
        label="Cross-sec. mean degree",
    )
    ax.plot(deg_df.index, mean_deg, color="steelblue", linewidth=1.0, alpha=0.6)

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, ticker in enumerate(available):
        series = deg_df[ticker].dropna()
        ax.plot(
            series.index,
            series.values,
            label=ticker,
            linewidth=1.5,
            color=colours[i % len(colours)],
        )

    edge_counts = network.snapshot_edge_counts()
    ax2 = ax.twinx()
    ax2.step(
        edge_counts.index,
        edge_counts.values,
        color="grey",
        linewidth=0.8,
        alpha=0.4,
        where="post",
        label="Total edges",
    )
    ax2.set_ylabel("Total edges in graph", color="grey", fontsize=9)
    ax2.tick_params(axis="y", colors="grey", labelsize=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlabel("Date")
    ax.set_ylabel("Node degree")
    ax.set_title(
        f"Network Degree Dynamics  [{type(network).__name__}, "
        f"window={network.window}, {network.graph_type}]",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graph neighbourhood comparison  (first vs last snapshot)
# ---------------------------------------------------------------------------

def _load_graph_json(path: "str | Path") -> dict:
    """Load a saved graph-snapshot JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _snapshot_to_nx(snap: dict) -> nx.Graph:
    """Reconstruct a NetworkX graph from a serialised snapshot dict."""
    G = nx.Graph()
    G.add_nodes_from(snap["nodes"])
    for e in snap["edges"]:
        G.add_edge(e["source"], e["target"], weight=e["weight"])
    return G


def _ego_subgraph(G: nx.Graph, center: str, k: int = 5) -> nx.Graph:
    """Extract the k-nearest-neighbour ego graph around *center*.

    Keeps only the *k* closest neighbours (by edge weight = distance)
    plus the center node.  Edges among neighbours are included if
    they exist in *G*.
    """
    if center not in G:
        return nx.Graph()
    neighbours = list(G.neighbors(center))
    nb_dist = [(nb, G[center][nb]["weight"]) for nb in neighbours]
    nb_dist.sort(key=lambda x: x[1])
    top_k = [nb for nb, _ in nb_dist[:k]]
    nodes = [center] + top_k
    return G.subgraph(nodes).copy()


def plot_neighbourhood_change(
    graph_json_path: "str | Path",
    focus_tickers: "list[str] | None" = None,
    k: int = 5,
    save_path: "str | None" = None,
):
    """
    For each focus ticker produce a **dedicated full-page figure** (22 × 11 in)
    comparing the ego network at the first and last graph snapshot side-by-side.

    One figure (and PNG) is emitted per ticker so every plot is large and
    legible.  When *save_path* is supplied it is used as a filename template:
    ``_{ticker}`` is inserted before the extension for each output file, e.g.
    ``neighbourhood_sqcorr_k5_AAPL.png``.

    Nodes that are neighbours in **both** snapshots are coloured blue;
    nodes present only in the early snapshot are orange; nodes only in the
    late snapshot are green.  Edge thickness is proportional to similarity
    (1 − distance).

    Parameters
    ----------
    graph_json_path : Path to a saved graph JSON (e.g. ``results/graphs/sqcorr_k5.json``).
    focus_tickers   : Tickers to plot.  Defaults to ``AAPL, NVDA, TSLA, MSFT``.
    k               : Number of nearest neighbours to show per ego graph.
    save_path       : Template path for saved PNGs; ``_{ticker}`` is inserted
                      before the ``.png`` extension for each ticker.
    """
    if focus_tickers is None:
        focus_tickers = ["AAPL", "NVDA", "TSLA", "MSFT"]

    data = _load_graph_json(graph_json_path)
    snaps = data["snapshots"]
    if len(snaps) < 2:
        print("[plot_neighbourhood_change] Need at least 2 snapshots.")
        return

    first_snap = snaps[0]
    last_snap  = snaps[-1]
    G_first = _snapshot_to_nx(first_snap)
    G_last  = _snapshot_to_nx(last_snap)

    first_date = first_snap["date"][:10]
    last_date  = last_snap["date"][:10]

    # Filter to tickers actually present in both graphs
    focus_tickers = [t for t in focus_tickers if t in G_first and t in G_last]
    if not focus_tickers:
        print("[plot_neighbourhood_change] No focus tickers found in both snapshots.")
        return

    net_class = data.get("network_class", "Network")
    k_val     = data.get("hyperparams", {}).get("k", k)
    title_suffix = f"{net_class}, k={k_val}"

    # Shared legend patches (repeated on every figure)
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="Focus ticker"),
        mpatches.Patch(color="#3498db", label="Neighbour (both snapshots)"),
        mpatches.Patch(color="#e67e22", label="Neighbour (first only)"),
        mpatches.Patch(color="#2ecc71", label="Neighbour (last only)"),
    ]

    for ticker in focus_tickers:
        ego_first = _ego_subgraph(G_first, ticker, k)
        ego_last  = _ego_subgraph(G_last,  ticker, k)

        nb_first = set(ego_first.nodes()) - {ticker}
        nb_last  = set(ego_last.nodes())  - {ticker}
        stable   = nb_first & nb_last
        lost     = nb_first - nb_last
        gained   = nb_last  - nb_first  # noqa: F841  (used implicitly via node_colors)

        # One large, dedicated figure per ticker
        fig, axes = plt.subplots(1, 2, figsize=(22, 11))

        for ax, (ego, snap_date, snap_label) in zip(axes, [
            (ego_first, first_date, "First"),
            (ego_last,  last_date,  "Last"),
        ]):
            if ego.number_of_nodes() == 0:
                ax.set_title(
                    f"{ticker} — {snap_label} ({snap_date})\n(not in graph)",
                    fontsize=14, fontweight="bold",
                )
                ax.axis("off")
                continue

            # Node colours
            node_colors = []
            for n in ego.nodes():
                if n == ticker:
                    node_colors.append("#e74c3c")   # red: focus
                elif n in stable:
                    node_colors.append("#3498db")   # blue: in both
                elif n in lost:
                    node_colors.append("#e67e22")   # orange: only in first
                else:
                    node_colors.append("#2ecc71")   # green: only in last

            # Edge widths (thicker = closer = lower weight/distance)
            edge_widths = [
                max(1.0, 6.0 * (1.0 - d.get("weight", 0.5)))
                for _, _, d in ego.edges(data=True)
            ]

            # Generous node spacing so labels never overlap
            pos = nx.spring_layout(ego, seed=42, k=3.5)

            nx.draw_networkx_nodes(
                ego, pos, ax=ax,
                node_color=node_colors,
                node_size=2200,
                edgecolors="black",
                linewidths=1.2,
            )
            nx.draw_networkx_labels(
                ego, pos, ax=ax,
                font_size=12, font_weight="bold",
            )
            nx.draw_networkx_edges(
                ego, pos, ax=ax,
                width=edge_widths,
                alpha=0.65,
                edge_color="grey",
            )
            edge_labels = {
                (u, v): f"{d.get('weight', 0):.2f}"
                for u, v, d in ego.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                ego, pos, edge_labels, ax=ax,
                font_size=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )

            ax.set_title(
                f"{ticker} — {snap_label} Snapshot ({snap_date})",
                fontsize=14, fontweight="bold", pad=12,
            )
            ax.axis("off")

        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=4,
            fontsize=11,
            framealpha=0.85,
        )
        fig.suptitle(
            f"Neighbourhood Evolution: {ticker}  [{title_suffix}]",
            fontsize=15, fontweight="bold", y=1.01,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 1])

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            ticker_path = p.parent / f"{p.stem}_{ticker}{p.suffix}"
            fig.savefig(ticker_path, dpi=150, bbox_inches="tight")

        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    repo_root   = Path(__file__).parent.parent
    results_dir = repo_root / "results"
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load fold-level metrics ──────────────────────────────────────────────
    bench_path = results_dir / "results_bench.json"
    if not bench_path.exists():
        print(f"ERROR: {bench_path} not found. Run evaluation/cross_val.py first.")
        sys.exit(1)

    metrics_df = pd.read_json(bench_path, orient="records")
    for _col in ("R2", "R2_log"):
        if _col in metrics_df.columns:
            metrics_df = metrics_df[metrics_df[_col] >= -1e6]

    # ── Load prediction store from saved CSVs ───────────────────────────────
    pred_dir = results_dir / "predictions_regression"
    pred_store: dict = {}
    if pred_dir.exists():
        for csv_path in sorted(pred_dir.glob("*_predictions.csv")):
            ticker = csv_path.stem.replace("_predictions", "")
            pred_store[ticker] = pd.read_csv(
                csv_path, index_col="Date", parse_dates=True
            )
    else:
        print(f"WARNING: {pred_dir} not found — skipping prediction plots.")

    sample_tickers = sorted(pred_store.keys())

    # ── 1. Per-ticker prediction plots ──────────────────────────────────────
    if pred_store:
        print(f"\nPlotting predictions for: {sample_tickers}")
        # Save raw-selection plots
        raw_dir = plots_dir / "predictions_raw"
        print(f"Writing raw-selection plots to {raw_dir}")
        plot_ticker_predictions(
            pred_store,
            metrics_df,
            sample_tickers,
            save_dir=str(raw_dir),
            use_log=False,
        )

        # Save log-selection plots
        log_dir = plots_dir / "predictions_log"
        print(f"Writing log-selection plots to {log_dir}")
        plot_ticker_predictions(
            pred_store,
            metrics_df,
            sample_tickers,
            save_dir=str(log_dir),
            use_log=True,
        )

    # ── 2. Graph neighbourhood change plots ─────────────────────────────────
    focus      = ["AAPL", "TSLA", "GOOG", "META", "MSFT", "NVDA", "NFLX", "AMZN"]
    graphs_dir = results_dir / "graphs"

    for fname, tag in [
        ("sqcorr_k5.json", "sqcorr"),
        ("pcorr_k5.json",  "pcorr"),
        ("mi_k5.json",     "mi"),
    ]:
        gpath = graphs_dir / fname
        if not gpath.exists():
            print(f"WARNING: {gpath} not found — skipping {tag} neighbourhood plot.")
            continue
        print(f"\nPlotting neighbourhood change for {tag} (k=5) ...")
        plot_neighbourhood_change(
            graph_json_path=gpath,
            focus_tickers=focus,
            k=5,
            save_path=str(plots_dir / f"neighbourhood_change_{tag}_k5.png"),
        )

    print(f"\nAll plots saved to {plots_dir}")