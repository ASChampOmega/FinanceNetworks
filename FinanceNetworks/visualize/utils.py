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


def coalesce_categories(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge k-variants into super-categories, keeping only the best
    (category, model) pair per super-category (by pct_R2_pos).

    For categories without a [k=N] suffix (e.g. HAR, ARIMA, GARCH) all rows
    are kept unchanged.  For categories *with* a [k=N] suffix only the rows
    of the winning (k, model) remain, relabelled to the super-category.

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

    # For each k-bearing super-category pick the best (Category, Model) by
    # pct_R2_pos, computed across tickers.
    per_ticker = (
        df[df["_super"].isin(k_supers)]
        .groupby(["Category", "Model", "Ticker"])["R2"]
        .mean()
        .reset_index()
    )
    pct = (
        per_ticker.groupby(["Category", "Model"])
        .agg(pct_R2_pos=("R2", lambda x: float((x > 0).mean())))
        .reset_index()
    )
    pct["_super"] = pct["Category"].map(_super_category)

    best_pairs: dict = {}   # {super_cat: (original_category, model_name)}
    for sc, grp in pct.groupby("_super"):
        idx = grp["pct_R2_pos"].idxmax()
        best_pairs[sc] = (grp.loc[idx, "Category"], grp.loc[idx, "Model"])

    # Keep only the winning rows and relabel Category -> super-category
    keep_mask = pd.Series(True, index=df.index)
    for sc, (orig_cat, best_model) in best_pairs.items():
        is_this_super = df["_super"] == sc
        is_winner = (df["Category"] == orig_cat) & (df["Model"] == best_model)
        keep_mask &= ~is_this_super | is_winner

    df = df[keep_mask].copy()
    df["Category"] = df["_super"]
    return df.drop(columns="_super")
