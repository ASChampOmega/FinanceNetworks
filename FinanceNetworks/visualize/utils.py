"""
visualize/utils.py
==================
Shared utilities for result post-processing.

Functions
---------
_super_category     : Strip [k=N] suffix from a category name.
coalesce_categories : Merge k-variant categories into super-categories,
                      keeping the best (k, model) pair by pct_R2_pos.
"""

import re
import pandas as pd


def _super_category(cat: str) -> str:
    """Strip '[k=N]' suffix to get the super-category name.

    'Network [k=3]'       -> 'Network'
    'PCorr Network [k=1]' -> 'PCorr Network'
    'ExpKernel [k=5]'     -> 'ExpKernel'
    'HAR'                 -> 'HAR'
    """
    return re.sub(r"\s*\[k=\d+\]$", "", cat)


def coalesce_categories(
    metrics_df: pd.DataFrame,
    metric_col: str = "R2",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Merge k-variants into super-categories, keeping **all models** from
    the best k value per super-category.

    For categories without a ``[k=N]`` suffix (e.g. HAR, ARIMA, GARCH) all
    rows are kept unchanged.  For categories *with* a ``[k=N]`` suffix the
    best k value is selected (by the best individual model's mean metric
    across tickers), then **every** model from that k is retained and
    relabelled to the super-category.

    Parameters
    ----------
    metrics_df       : DataFrame with at least Category, Ticker, Model,
                       and *metric_col*.
    metric_col       : Column used to rank models when selecting the best k
                       (default ``"R2"``).  Use ``"ROC_AUC"`` for
                       classification.
    higher_is_better : ``True`` when a larger value is better (R2, ROC-AUC);
                       ``False`` when lower is better (RMSE).

    Returns a copy of metrics_df with the Category column updated.
    """
    df = metrics_df.copy()
    df["_super"] = df["Category"].map(_super_category)

    # Identify super-categories that actually span multiple k values
    k_supers = (
        df.groupby("_super")["Category"]
        .nunique()
        .loc[lambda x: x > 1]
        .index.tolist()
    )
    if not k_supers:
        # Nothing to coalesce — just drop the helper column
        df["Category"] = df["_super"]
        return df.drop(columns="_super")

    # Per-ticker mean of the metric for each (Category, Model)
    per_ticker = (
        df[df["_super"].isin(k_supers)]
        .groupby(["Category", "Ticker", "Model"])[metric_col]
        .mean()
        .reset_index()
    )
    per_model = (
        per_ticker.groupby(["Category", "Model"])[metric_col]
        .mean()
        .reset_index()
    )

    # For each Category (== a specific k), find the best model's score
    agg_fn = "max" if higher_is_better else "min"
    best_per_cat = (
        per_model.groupby("Category")[metric_col]
        .agg(agg_fn)
        .reset_index()
    )
    best_per_cat["_super"] = best_per_cat["Category"].map(_super_category)

    # For each super-category, pick the k whose best model wins
    best_k_cat: dict = {}  # {super_cat: original_category}
    for sc, grp in best_per_cat.groupby("_super"):
        if higher_is_better:
            idx = grp[metric_col].idxmax()
        else:
            idx = grp[metric_col].idxmin()
        best_k_cat[sc] = grp.loc[idx, "Category"]

    # Keep ALL rows from the winning k category per super-category
    keep_mask = pd.Series(True, index=df.index)
    for sc, orig_cat in best_k_cat.items():
        is_this_super = df["_super"] == sc
        is_best_k = df["Category"] == orig_cat
        keep_mask &= ~is_this_super | is_best_k

    df = df[keep_mask].copy()
    df["Category"] = df["_super"]
    return df.drop(columns="_super")
