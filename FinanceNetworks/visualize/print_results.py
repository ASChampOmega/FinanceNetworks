"""
visualize/print_results.py
==========================
Result aggregation and console-printing utilities.

Functions
---------
summarize_benchmarks          : Aggregate fold-level metrics across tickers.
print_summary                 : Print the full summary table.
print_best_per_category       : Print the single best model per category.
print_per_ticker_tables       : Print per-ticker metrics for sample tickers.
print_summary_excluding_outliers : Reprint summary after dropping bad tickers.
load_and_print_results        : Load saved JSON results and reprint everything.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from visualize.utils import coalesce_categories


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_benchmarks(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics first per (Category, Ticker, Model), then across tickers."""
    per_ticker = (
        metrics_df.groupby(["Category", "Ticker", "Model"])
        .agg(
            R2=("R2", "mean"),
            RMSE=("RMSE", "mean"),
            MAE=("MAE", "mean"),
            R2_log=("R2_log", "mean"),
            RMSE_log=("RMSE_log", "mean"),
            MAE_log=("MAE_log", "mean"),
        )
        .reset_index()
    )

    summary = (
        per_ticker.groupby(["Category", "Model"])
        .agg(
            mean_R2=("R2", "mean"),
            median_R2=("R2", "median"),
            pct_R2_pos=("R2", lambda x: float((x > 0).mean())),
            mean_RMSE=("RMSE", "mean"),
            median_RMSE=("RMSE", "median"),
            mean_MAE=("MAE", "mean"),
            n_tickers=("Ticker", "nunique"),
        )
        .sort_values("pct_R2_pos", ascending=False)
    )
    return summary


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_PRINT_COLS = ["mean_R2", "median_R2", "pct_R2_pos", "mean_RMSE", "median_RMSE"]
_FMT = {"display.max_columns": None, "display.width": None,
        "display.float_format": "{:.4f}".format}    


def print_summary(summary: pd.DataFrame, title: str = "Summary") -> None:
    """Print the full summary table (sorted by pct_R2_pos) without column truncation."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    cols = [c for c in _PRINT_COLS if c in summary.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        print(summary[cols].to_string())


def print_best_per_category(summary: pd.DataFrame) -> None:
    """Print a compact table showing the single best model per category (by pct_R2_pos)."""
    print(f"\n{'=' * 80}")
    print("  Best Model per Category (by pct_R2_pos)")
    print(f"{'=' * 80}")
    best_rows = []
    for cat in summary.index.get_level_values("Category").unique():
        cat_df = summary.loc[cat]
        best_model = cat_df["pct_R2_pos"].idxmax()
        row = cat_df.loc[best_model]
        best_rows.append({"Category": cat, "Model": best_model, **row.to_dict()})
    best_df = pd.DataFrame(best_rows).set_index(["Category", "Model"])
    cols = [c for c in _PRINT_COLS if c in best_df.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        print(best_df[cols].to_string())


def print_per_ticker_tables(
    metrics_df: pd.DataFrame,
    tickers: List[str],
) -> None:
    """Print per-ticker R2 / RMSE / MAE for the best model per category (by pct_R2_pos)."""
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
    best_models: dict = {}
    for cat, grp in pct_pos.groupby("Category"):
        best_models[cat] = grp.loc[grp["pct_R2_pos"].idxmax(), "Model"]

    for ticker in tickers:
        tm = metrics_df[metrics_df["Ticker"] == ticker]
        if tm.empty:
            continue
        print(f"\n{'─' * 60}")
        print(f"  {ticker}")
        print(f"{'─' * 60}")
        rows = []
        for cat, mn in sorted(best_models.items()):
            msk = (tm["Category"] == cat) & (tm["Model"] == mn)
            sub = tm[msk]
            if sub.empty:
                continue
            rows.append({
                "Category": cat,
                "Model": mn,
                "R2": sub["R2"].mean(),
                "RMSE": sub["RMSE"].mean(),
                "MAE": sub["MAE"].mean(),
            })
        if rows:
            tdf = pd.DataFrame(rows).set_index(["Category", "Model"])
            with pd.option_context("display.float_format", "{:.4f}".format,
                                   "display.width", None):
                print(tdf.to_string())


def print_summary_excluding_outliers(
    metrics_df: pd.DataFrame,
    r2_threshold: float = -1.0,
) -> None:
    """Re-aggregate and reprint summary after dropping tickers whose mean R2 < threshold."""
    per_ticker = (
        metrics_df.groupby(["Category", "Ticker", "Model"])
        .agg(R2=("R2", "mean"))
        .reset_index()
    )
    bad_tickers = per_ticker.loc[
        per_ticker.groupby("Ticker")["R2"].transform("mean") < r2_threshold,
        "Ticker",
    ].unique()

    if len(bad_tickers) == 0:
        print(f"\n  (No outlier tickers — all tickers have mean R2 >= {r2_threshold:.2f})")
        return

    print(f"\n  Excluding {len(bad_tickers)} outlier tickers "
          f"(ticker-mean R2 < {r2_threshold}): {sorted(bad_tickers)}")
    clean = metrics_df[~metrics_df["Ticker"].isin(bad_tickers)]
    print_summary(summarize_benchmarks(clean), title="Summary (outlier tickers removed)")


# ---------------------------------------------------------------------------
# Load-from-disk entry point
# ---------------------------------------------------------------------------

def load_and_print_results(
    results_dir: "str | Path" = None,
    sample_tickers: "Optional[List[str]]" = None,
    r2_threshold: float = -1.0,
    plot: bool = False,
) -> None:
    """
    Load saved JSON results and reprint all summary tables.

    Categories with ``[k=N]`` suffixes are coalesced into super-categories
    (e.g. ``'Network [k=1/3/5]'`` → ``'Network'``) by selecting the best
    ``(k, model)`` pair per super-category by pct_R2_pos.

    Parameters
    ----------
    results_dir    : Folder containing ``results_bench.json``.
                     Defaults to ``<repo_root>/FinanceNetworks/results``.
    sample_tickers : Tickers for per-ticker tables.
                     Defaults to ``["AAPL", "TSLA", "GOOG", "META", "MSFT"]``.
    r2_threshold   : Drop tickers whose cross-model mean R2 < this value.
    plot           : If True, regenerate and save the summary bar chart.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"
    results_dir = Path(results_dir)

    if sample_tickers is None:
        sample_tickers = ["AAPL", "TSLA", "GOOG", "META", "MSFT"]

    raw_path = results_dir / "results_bench.json"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run main() first.")
        return

    print(f"Loading fold-level metrics from {raw_path} ...")
    metrics_df = pd.read_json(raw_path, orient="records")
    metrics_df = coalesce_categories(metrics_df)
    summary = summarize_benchmarks(metrics_df)

    print_summary(summary, title="Full Summary (from saved results)")
    print_best_per_category(summary)
    print_summary_excluding_outliers(metrics_df, r2_threshold=r2_threshold)
    print_per_ticker_tables(metrics_df, sample_tickers)
    print(f"\n  (Loaded {len(metrics_df)} rows from {raw_path})")

    if plot:
        from visualize.plot_model_results import plot_summary_metrics
        plot_summary_metrics(
            summary,
            save_path=str(results_dir / "summary_metrics.png"),
        )
