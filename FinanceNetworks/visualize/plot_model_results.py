import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_ticker_predictions(
    pred_store: dict,
    sample_tickers: list,
    save_dir: "str | None" = None,
):
    """
    For each ticker in sample_tickers, plot true Y_fwd against each model's forecast.

    Parameters
    ----------
    pred_store     : output of run_benchmarks_multi_fold {ticker: DataFrame with Y_true + model cols}
    sample_tickers : list of tickers to plot
    save_dir       : if given, saves each figure as <save_dir>/<ticker>_predictions.png
    """
    for ticker in sample_tickers:
        if ticker not in pred_store:
            continue

        df = pred_store[ticker]
        # Keep only rows that appear in at least one test fold (model cols not all NaN)
        model_cols = [c for c in df.columns if c != "Y_true"]
        df = df.dropna(subset=model_cols, how="all")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df["Y_true"], label="True", color="black", linewidth=1.2, alpha=0.85, zorder=5)

        for col in [c for c in df.columns if c != "Y_true"]:
            ax.plot(df.index, df[col], label=col, linewidth=1.0, alpha=0.80)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_title(ticker, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Realized Variance (Y_fwd)")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()

        if save_dir is not None:
            fig.savefig(f"{save_dir}/{ticker}_predictions.png", dpi=150)

        plt.show()


def plot_summary_metrics(
    summary_df: pd.DataFrame,
    save_path: "str | None" = None,
):
    """
    Side-by-side horizontal bar charts of mean_R2 and mean_R2_log for each model.
    """
    metrics = [("mean_R2", "Mean R² (original scale)"), ("mean_R2_log", "Mean R² (log scale)")]
    # Only plot columns that exist (for backwards compat)
    metrics = [(col, label) for col, label in metrics if col in summary_df.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), max(3, len(summary_df) * 0.7)))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        data = summary_df[col].sort_values(ascending=True)
        bars = ax.barh(data.index, data.values, color="steelblue", edgecolor="white")
        for bar, val in zip(bars, data.values):
            ax.text(
                val + 0.003 if val >= 0 else val - 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9,
            )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(label)
        ax.set_title(label, fontsize=12, fontweight="bold")

    fig.suptitle("Model Comparison Across Tickers & Folds", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


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
    deg_df = network.snapshot_degrees()  # (n_snapshots x n_tickers)

    if deg_df.empty:
        print("[plot_network_degrees] No snapshot data available.")
        return

    # Limit to tickers that have at least some data
    available = [t for t in (sample_tickers or []) if t in deg_df.columns]
    if not available:
        available = deg_df.columns[deg_df.notna().any()].tolist()[:5]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Background: cross-sectional mean degree
    mean_deg = deg_df.mean(axis=1)
    ax.fill_between(
        deg_df.index,
        mean_deg,
        alpha=0.12,
        color="steelblue",
        label="Cross-sec. mean degree",
    )
    ax.plot(deg_df.index, mean_deg, color="steelblue", linewidth=1.0, alpha=0.6)

    # Foreground: per-ticker degree traces
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

    # Annotate edge counts on secondary axis
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
