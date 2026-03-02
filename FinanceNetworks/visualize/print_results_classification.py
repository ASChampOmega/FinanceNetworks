"""
visualize/print_results_classification.py
==========================================
Result aggregation and console-printing utilities for the classification task.

Primary evaluation metric: mean ROC-AUC across tickers (robust to class imbalance).
Full metric set: F1, Precision, Recall, ROC-AUC, Weighted Recall, Accuracy.

Weighted Recall weights each true spike by log(1 + Y_fwd) so that missing
a large volatility event is penalised more than missing a small one.

Functions
---------
summarize_classification          : Aggregate fold-level metrics across tickers.
print_classification_summary      : Print the full summary table.
print_best_classifier             : Print the single best model per category.
print_overall_best                : Print the single globally best model.
print_per_ticker_classification   : Print per-ticker metrics for sample tickers.
load_and_print_classification_results : Load saved JSON results and reprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_classification(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate fold-level classification metrics.

    Two-stage aggregation:
      1. Average across folds per (Category, Ticker, Model).
      2. Summarise across tickers per (Category, Model).

    Returns a MultiIndex DataFrame sorted by mean_ROC_AUC descending.
    """
    agg_cols = {
        "F1":              ("F1",              "mean"),
        "Precision":       ("Precision",       "mean"),
        "Recall":          ("Recall",          "mean"),
        "ROC_AUC":         ("ROC_AUC",         "mean"),
        "Weighted_Recall": ("Weighted_Recall", "mean"),
        "Accuracy":        ("Accuracy",        "mean"),
    }
    per_ticker = (
        metrics_df.groupby(["Category", "Ticker", "Model"])
        .agg(**agg_cols)
        .reset_index()
    )

    summary = (
        per_ticker.groupby(["Category", "Model"])
        .agg(
            mean_F1=("F1",              "mean"),
            mean_Precision=("Precision",       "mean"),
            mean_Recall=("Recall",          "mean"),
            mean_ROC_AUC=("ROC_AUC",         "mean"),
            mean_Weighted_Recall=("Weighted_Recall", "mean"),
            mean_Accuracy=("Accuracy",        "mean"),
            n_tickers=("Ticker",         "nunique"),
        )
        .sort_values("mean_ROC_AUC", ascending=False)
    )
    return summary


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_PRINT_COLS = [
    "mean_ROC_AUC",
    "mean_Precision",
    "mean_Recall",
    "mean_F1",
    "mean_Weighted_Recall",
    "mean_Accuracy",
    "n_tickers",
]

_FMT = {
    "display.max_columns": None,
    "display.width": None,
    "display.float_format": "{:.4f}".format,
}


def print_classification_summary(
    summary: pd.DataFrame,
    title: str = "Classification Summary",
) -> None:
    """Print the full classification summary table sorted by mean_ROC_AUC."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    cols = [c for c in _PRINT_COLS if c in summary.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        print(summary[cols].to_string())


def print_best_classifier(summary: pd.DataFrame) -> None:
    """
    Print a compact table showing the single best classifier per category
    (chosen by mean ROC_AUC).
    """
    print(f"\n{'=' * 80}")
    print("  Best Classifier per Category (by mean ROC_AUC)")
    print(f"{'=' * 80}")
    best_rows = []
    for cat in summary.index.get_level_values("Category").unique():
        cat_df = summary.loc[cat]
        best_model = cat_df["mean_ROC_AUC"].idxmax()
        row = cat_df.loc[best_model]
        best_rows.append({"Category": cat, "Model": best_model, **row.to_dict()})
    best_df = pd.DataFrame(best_rows).set_index(["Category", "Model"])
    cols = [c for c in _PRINT_COLS if c in best_df.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        print(best_df[cols].to_string())


def print_overall_best(summary: pd.DataFrame) -> None:
    """
    Print the single globally best model across all categories
    (chosen by mean ROC_AUC across all tickers).
    """
    print(f"\n{'=' * 80}")
    print("  Overall Best Classifier (by mean ROC_AUC across all tickers)")
    print(f"{'=' * 80}")
    best_idx = summary["mean_ROC_AUC"].idxmax()
    best_row = summary.loc[[best_idx]]
    cols = [c for c in _PRINT_COLS if c in best_row.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        print(best_row[cols].to_string())


def print_per_ticker_classification(
    metrics_df: pd.DataFrame,
    tickers: List[str],
) -> None:
    """
    Print per-ticker F1 / Precision / Recall / ROC-AUC / Weighted_Recall /
    Accuracy for the best model per category (chosen by mean F1 across
    all tickers).
    """
    # Identify the best model per category by cross-ticker mean F1
    per_ticker = (
        metrics_df.groupby(["Category", "Ticker", "Model"])
        .agg(F1=("F1", "mean"))
        .reset_index()
    )
    mean_f1 = (
        per_ticker.groupby(["Category", "Model"])
        .agg(mean_F1=("F1", "mean"))
        .reset_index()
    )
    best_models: dict = {}
    for cat, grp in mean_f1.groupby("Category"):
        best_models[cat] = grp.loc[grp["mean_F1"].idxmax(), "Model"]

    # Prefer ROC_AUC first when displaying per-ticker metrics
    _ticker_cols = ["ROC_AUC", "Precision", "Recall", "F1", "Weighted_Recall", "Accuracy"]

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
            row = {"Category": cat, "Model": mn}
            for col in _ticker_cols:
                if col in sub.columns:
                    row[col] = sub[col].mean()
            rows.append(row)
        if rows:
            tdf = pd.DataFrame(rows).set_index(["Category", "Model"])
            display_cols = [c for c in _ticker_cols if c in tdf.columns]
            with pd.option_context(
                "display.float_format", "{:.4f}".format,
                "display.width", None,
            ):
                print(tdf[display_cols].to_string())
            # Print the single best model for this ticker (prefer ROC_AUC)
            if "ROC_AUC" in tdf.columns:
                best_idx = tdf["ROC_AUC"].idxmax()
                best_score = tdf["ROC_AUC"].max()
                score_name = "ROC_AUC"
            elif "F1" in tdf.columns:
                best_idx = tdf["F1"].idxmax()
                best_score = tdf["F1"].max()
                score_name = "F1"
            else:
                print(f"\nBest Model for {ticker}: (no ROC_AUC or F1 available)")
                continue
            # best_idx is a (Category, Model) tuple because of the MultiIndex
            try:
                best_category, best_model = best_idx
            except Exception:
                best_category, best_model = (None, best_idx)
            print(f"\nBest Model for {ticker}: {best_model} (Category: {best_category}, {score_name}: {best_score:.4f})")


# ---------------------------------------------------------------------------
# Load-from-disk entry point
# ---------------------------------------------------------------------------

def load_and_print_classification_results(
    results_dir: "str | Path" = None,
    sample_tickers: "Optional[List[str]]" = None,
) -> None:
    """
    Load saved classification JSON results and reprint all summary tables.

    Parameters
    ----------
    results_dir    : Folder containing ``classification_results.json``.
                     Defaults to ``<repo_root>/FinanceNetworks/results``.
    sample_tickers : Tickers for per-ticker tables.
                     Defaults to ``["AAPL", "TSLA", "GOOG", "META", "MSFT"]``.
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results"
    results_dir = Path(results_dir)

    if sample_tickers is None:
        sample_tickers = ["AAPL", "TSLA", "GOOG", "META", "MSFT"]

    raw_path = results_dir / "classification_results.json"
    if not raw_path.exists():
        print(
            f"ERROR: {raw_path} not found. "
            "Run main() in evaluation/classification.py first."
        )
        return

    print(f"Loading classification metrics from {raw_path} ...")
    metrics_df = pd.read_json(raw_path, orient="records")
    summary = summarize_classification(metrics_df)

    print_classification_summary(summary, title="Classification Summary (from saved results)")
    print_best_classifier(summary)
    print_overall_best(summary)
    print_per_ticker_classification(metrics_df, sample_tickers)
    print(f"\n  (Loaded {len(metrics_df)} rows from {raw_path})")
