import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def plot_ticker_predictions(
    pred_store: dict,
    metrics_df: pd.DataFrame,
    sample_tickers: list,
    save_dir: "str | None" = None,
):
    """
    For each ticker in sample_tickers, plot true Y_fwd against ONE model per
    category -- the best model in that category by mean R2 across tickers.

    This keeps the legend readable regardless of how many model variants exist:
    only one curve per category (HAR, ARIMA, GARCH, Network [k=1], ...) is shown.

    Network categories are an exception: ALL network-category models are plotted
    since those are the proposed models and comparing them is the key analysis.

    Parameters
    ----------
    pred_store   : {ticker: DataFrame} with Y_true + one column per model name.
    metrics_df   : output of run_benchmarks_multi_fold (has Category, Model, R2 cols).
    sample_tickers: tickers to plot.
    save_dir     : if given, saves each figure as <save_dir>/<ticker>_predictions.png.
    """
    # ── Pick best model per category (by pct_R2_pos across all tickers) ──
    cat_best: dict = {}   # {category: [model_name]}
    per_ticker_r2 = (
        metrics_df.groupby(["Category", "Ticker", "Model"])["R2"]
        .mean()
        .reset_index()
    )
    pct_pos = (
        per_ticker_r2.groupby(["Category", "Model"])
        .agg(pct_R2_pos=("R2", lambda x: float((x > 0).mean())))
        .reset_index()
    )
    for cat, grp in pct_pos.groupby("Category"):
        best_model = grp.loc[grp["pct_R2_pos"].idxmax(), "Model"]
        cat_best[cat] = [best_model]

    for ticker in sample_tickers:
        if ticker not in pred_store:
            continue

        df = pred_store[ticker]
        model_cols = [c for c in df.columns if c != "Y_true"]
        df = df.dropna(subset=model_cols, how="all")

        # Collect the columns to plot: best-per-category for baselines,
        # all models for network categories.
        cols_to_plot = []
        for cat, model_names in sorted(cat_best.items()):
            for mn in model_names:
                col_key = f"[{cat}] {mn}"
                if col_key in df.columns:
                    cols_to_plot.append((cat, mn, col_key))

        n_curves = len(cols_to_plot)
        fig_h = max(5, 3 + n_curves * 0.35)
        fig, ax = plt.subplots(figsize=(14, fig_h))

        ax.plot(
            df.index, df["Y_true"],
            label="True Y_fwd",
            color="black",
            linewidth=1.4,
            alpha=0.9,
            zorder=5,
        )

        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (cat, model_name, col_key) in enumerate(cols_to_plot):
            label = f"[{cat}] {model_name}"
            ax.plot(
                df.index,
                df[col_key],
                label=label,
                linewidth=1.0,
                alpha=0.80,
                color=colours[i % len(colours)],
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_title(ticker, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Realized Variance (Y_fwd)")
        ax.legend(
            loc="upper left",
            fontsize=8,
            ncol=2,
            framealpha=0.7,
        )
        fig.tight_layout()

        if save_dir is not None:
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

    fig, ax = plt.subplots(figsize=(14, 6))

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