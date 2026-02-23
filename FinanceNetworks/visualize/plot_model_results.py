import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


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
