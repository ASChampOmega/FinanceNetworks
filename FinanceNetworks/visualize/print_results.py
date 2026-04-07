"""
visualize/print_results.py
==========================
Result aggregation and console-printing utilities.

Functions
---------
clip_extreme_metrics                  : Drop fold-level rows with numerical R2 blowups.
summarize_benchmarks                  : Aggregate fold-level metrics across tickers.
print_summary                         : Print the full summary table.
print_best_per_category               : Print the single best model per category.
print_compact_leaderboard             : Ultra-short leaderboard (best per category).
print_per_ticker_tables               : Print per-ticker metrics for sample tickers.
print_per_ticker_network_vs_baseline  : Best-network vs best-baseline for each ticker.
print_k_breakdown                     : Side-by-side comparison across all available k values.
print_weighting_scheme_breakdown      : IDW vs Exp vs LearnedWeight comparison.
print_wilcoxon_best_network_vs_baseline : Wilcoxon signed-rank test (network vs baseline).
print_summary_excluding_outliers      : Reprint summary after dropping bad tickers.
load_and_print_results                : Load saved JSON results and reprint everything.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from scipy.stats import wilcoxon

from visualize.utils import coalesce_categories

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_CATEGORIES: frozenset = frozenset({"HAR", "ARIMA", "GARCH", "RegimeSwitching"})


def _scale_metric_columns_for_display(
    metrics_df: pd.DataFrame,
    raw_metric_scale: float = 1.0,
) -> pd.DataFrame:
    """Return a copy with raw error metrics scaled for display only.

    This is intended for cases like the Oxford-Man index pipeline where the
    target is stored in raw variance units, making RMSE/MAE values look tiny
    next to the stock pipeline's percent-squared units. Scaling here does not
    affect any saved results or model ranking because it multiplies all raw
    error metrics by the same constant.
    """
    if raw_metric_scale == 1.0:
        return metrics_df

    scaled = metrics_df.copy()
    for col in ("RMSE", "MAE"):
        if col in scaled.columns:
            scaled[col] = scaled[col] * raw_metric_scale
    return scaled


# ---------------------------------------------------------------------------
# Extreme-value filter
# ---------------------------------------------------------------------------

def clip_extreme_metrics(
    metrics_df: pd.DataFrame,
    r2_floor: float = -1e6,
) -> pd.DataFrame:
    """Drop fold-level rows where R2 or R2_log is a clear numerical artifact.

    Values below *r2_floor* (default -1e6) indicate numerical blowups and are
    excluded so they don't drag aggregate means to meaningless extremes.
    """
    mask = pd.Series(True, index=metrics_df.index)
    for col in ("R2", "R2_log"):
        if col in metrics_df.columns:
            mask &= metrics_df[col] >= r2_floor
    dropped = int((~mask).sum())
    if dropped:
        print(
            f"  [filter] Removed {dropped} fold-level rows with R2/R2_log "
            f"< {r2_floor:.0e} (numerical artifacts)."
        )
    return metrics_df[mask].copy()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_benchmarks(
    metrics_df: pd.DataFrame,
    use_log: bool = True,
    model_r2_floor: float = -100.0,
    selection: str = "r2",
) -> pd.DataFrame:
    """Aggregate fold-level metrics first per (Category, Ticker, Model), then across tickers.

    Parameters
    ----------
    use_log : bool
        When True use log-scale metrics; otherwise raw metrics.
    model_r2_floor : float
        (Category, Model) pairs whose ticker-mean R2 (chosen by *use_log*) falls
        below this value are silently dropped as degenerate.
    selection : str
        Column used to sort / rank models. One of:
        ``'r2'`` (default) – sort by pct_R2_pos (higher is better);
        ``'mean_rmse'``    – sort by mean RMSE (lower is better);
        ``'median_rmse'``  – sort by median RMSE (lower is better).
    """
    r2_col = "R2_log" if use_log else "R2"

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

    # Drop (Category, Model) pairs with absurdly low ticker-mean R2
    model_means = (
        per_ticker.groupby(["Category", "Model"])[r2_col]
        .mean()
        .reset_index()
        .rename(columns={r2_col: "__mean_r2"})
    )
    good = model_means[model_means["__mean_r2"] >= model_r2_floor][["Category", "Model"]]
    n_before = per_ticker[["Category", "Model"]].drop_duplicates().shape[0]
    per_ticker = per_ticker.merge(good, on=["Category", "Model"], how="inner")
    n_after = per_ticker[["Category", "Model"]].drop_duplicates().shape[0]
    if n_before > n_after:
        print(
            f"  [filter] Dropped {n_before - n_after} (Category, Model) pairs "
            f"with mean {r2_col} < {model_r2_floor}."
        )

    if selection == "mean_rmse":
        sort_col      = "mean_RMSE_log" if use_log else "mean_RMSE"
        sort_ascending = True
    elif selection == "median_rmse":
        sort_col      = "median_RMSE_log" if use_log else "median_RMSE"
        sort_ascending = True
    else:  # "r2"
        sort_col      = "pct_R2_log_pos" if use_log else "pct_R2_pos"
        sort_ascending = False
    summary = (
        per_ticker.groupby(["Category", "Model"])
        .agg(
            mean_R2=("R2", "mean"),
            median_R2=("R2", "median"),
            pct_R2_pos=("R2", lambda x: float((x > 0).mean())),
            mean_RMSE=("RMSE", "mean"),
            median_RMSE=("RMSE", "median"),
            mean_MAE=("MAE", "mean"),
            mean_R2_log=("R2_log", "mean"),
            median_R2_log=("R2_log", "median"),
            pct_R2_log_pos=("R2_log", lambda x: float((x > 0).mean())),
            mean_RMSE_log=("RMSE_log", "mean"),
            median_RMSE_log=("RMSE_log", "median"),
            mean_MAE_log=("MAE_log", "mean"),
            n_tickers=("Ticker", "nunique"),
        )
        .sort_values(sort_col, ascending=sort_ascending)
    )
    return summary


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_PRINT_COLS = [
    "mean_R2", "median_R2", "pct_R2_pos", "mean_RMSE", "median_RMSE",
    "mean_R2_log", "median_R2_log", "pct_R2_log_pos", "mean_RMSE_log", "median_RMSE_log",
]
_FMT = {"display.max_columns": None, "display.width": 220,
        "display.max_colwidth": 60,
        "display.float_format": "{:.4f}".format}    


def print_summary(summary: pd.DataFrame, title: str = "Summary") -> None:
    """Print the full summary table (sorted by pct_R2_pos) without column truncation."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    cols = [c for c in _PRINT_COLS if c in summary.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        print(summary[cols].to_string())


def print_best_per_category(summary: pd.DataFrame, use_log: bool = True, selection: str = "r2") -> None:
    """Print a compact table showing the single best model per category."""
    if selection == "mean_rmse":
        sel_col = "mean_RMSE_log" if use_log else "mean_RMSE"
        sel_label = sel_col
        def _pick_best(cat_df):
            return cat_df[sel_col].idxmin()
    elif selection == "median_rmse":
        sel_col = "median_RMSE_log" if use_log else "median_RMSE"
        sel_label = sel_col
        def _pick_best(cat_df):
            return cat_df[sel_col].idxmin()
    else:  # "r2"
        sel_col = "pct_R2_log_pos" if use_log else "pct_R2_pos"
        sel_label = sel_col
        def _pick_best(cat_df):
            return cat_df[sel_col].idxmax()
    print(f"\n{'=' * 80}")
    print(f"  Best Model per Category (by {sel_label})")
    print(f"{'=' * 80}")
    best_rows = []
    for cat in summary.index.get_level_values("Category").unique():
        cat_df = summary.loc[cat]
        best_model = _pick_best(cat_df)
        row = cat_df.loc[best_model]
        best_rows.append({"Category": cat, "Model": best_model, **row.to_dict()})
    best_df = pd.DataFrame(best_rows).set_index(["Category", "Model"])
    cols = [c for c in _PRINT_COLS if c in best_df.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        sort_asc = selection in ("mean_rmse", "median_rmse")
        print(best_df[cols].sort_values(by=sel_col, ascending=sort_asc).to_string())


def print_per_ticker_tables(
    metrics_df: pd.DataFrame,
    tickers: List[str],
    use_log: bool = True,
    selection: str = "r2",
) -> None:
    """Print per-ticker R2 / RMSE / MAE for the best model per category."""
    rmse_sel = "RMSE_log" if use_log else "RMSE"
    r2_sel   = "R2_log"   if use_log else "R2"

    per_ticker = (
        metrics_df.groupby(["Category", "Ticker", "Model"])[[r2_sel, rmse_sel]]
        .mean()
        .reset_index()
    )

    best_models: dict = {}
    if selection == "mean_rmse":
        agg = per_ticker.groupby(["Category", "Model"])[rmse_sel].mean().reset_index()
        for cat, grp in agg.groupby("Category"):
            best_models[cat] = grp.loc[grp[rmse_sel].idxmin(), "Model"]
    elif selection == "median_rmse":
        agg = per_ticker.groupby(["Category", "Model"])[rmse_sel].median().reset_index()
        for cat, grp in agg.groupby("Category"):
            best_models[cat] = grp.loc[grp[rmse_sel].idxmin(), "Model"]
    else:  # "r2"
        pct_pos = (
            per_ticker.groupby(["Category", "Model"])
            .agg(pct_pos=(r2_sel, lambda x: float((x > 0).mean())))
            .reset_index()
        )
        for cat, grp in pct_pos.groupby("Category"):
            best_models[cat] = grp.loc[grp["pct_pos"].idxmax(), "Model"]

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
            row = {
                "Category": cat,
                "Model": mn,
                "R2": sub["R2"].mean(),
                "RMSE": sub["RMSE"].mean(),
                "MAE": sub["MAE"].mean(),
            }
            if "R2_log" in sub.columns:
                row["R2_log"] = sub["R2_log"].mean()
                row["RMSE_log"] = sub["RMSE_log"].mean()
                row["MAE_log"] = sub["MAE_log"].mean()
            rows.append(row)
        if rows:
            tdf = pd.DataFrame(rows).set_index(["Category", "Model"])
            with pd.option_context("display.float_format", "{:.4f}".format,
                                   "display.width", None):
                print(tdf.to_string())


def print_compact_leaderboard(summary: pd.DataFrame, use_log: bool = True, selection: str = "r2") -> None:
    """Ultra-short best-per-category table with only the key metrics."""
    rmse_key = "mean_RMSE_log" if use_log else "mean_RMSE"
    r2_key   = "mean_R2_log"   if use_log else "mean_R2"
    pct_key  = "pct_R2_log_pos" if use_log else "pct_R2_pos"
    rmse_lbl = "RMSE_log" if use_log else "RMSE"
    r2_lbl   = "R2_log"   if use_log else "R2"
    metric_label = "log metrics" if use_log else "raw metrics"
    if selection == "mean_rmse":
        sel_col = rmse_key
        def _pick_best(cat_df): return cat_df[sel_col].idxmin()
    elif selection == "median_rmse":
        sel_col = "median_RMSE_log" if use_log else "median_RMSE"
        def _pick_best(cat_df): return cat_df[sel_col].idxmin()
    else:  # "r2"
        sel_col = pct_key
        def _pick_best(cat_df): return cat_df[sel_col].idxmax()
    print(f"\n{'=' * 80}")
    print(f"  Leaderboard (best model per category, {metric_label}, selection={selection})")
    print(f"{'=' * 80}")
    rows = []
    for cat in summary.index.get_level_values("Category").unique():
        cat_df = summary.loc[cat]
        best_model = _pick_best(cat_df)
        r = cat_df.loc[best_model]
        rows.append({
            "Category": cat,
            "Model":    best_model,
            rmse_lbl:   r.get(rmse_key, float("nan")),
            r2_lbl:     r.get(r2_key,   float("nan")),
            "%R2>0":    r.get(pct_key,  float("nan")),
        })
    lb = pd.DataFrame(rows).set_index(["Category", "Model"])
    if selection == "mean_rmse":
        lb = lb.sort_values(rmse_lbl, ascending=True)
    elif selection == "median_rmse":
        lb = lb.sort_values("median_" + rmse_lbl if "median_" + rmse_lbl in lb.columns else rmse_lbl, ascending=True)
    else:  # r2
        lb = lb.sort_values("%R2>0", ascending=False)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 220, "display.max_colwidth", 60):
        print(lb.to_string())


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


def print_wilcoxon_best_network_vs_baseline(
    metrics_df: pd.DataFrame,
    use_log: bool = True,
) -> None:
    """
    Compare the best network model against the best baseline using paired
    per-ticker RMSE (log or raw depending on *use_log*).

    Model selection uses the lowest mean per-ticker RMSE (after averaging over
    folds). The Wilcoxon signed-rank test is run on the paired ticker-level
    RMSE series for those two winning models.
    """
    rmse_col     = "RMSE_log" if use_log else "RMSE"
    metric_label = "log metrics" if use_log else "raw metrics"

    if metrics_df.empty or rmse_col not in metrics_df.columns:
        print(f"\n  (Wilcoxon test skipped: {rmse_col} column not available.)")
        return

    per_ticker = (
        metrics_df
        .groupby(["Category", "Ticker", "Model"])
        .agg(**{rmse_col: (rmse_col, "mean")})
        .reset_index()
    )

    baseline_df = per_ticker[per_ticker["Category"].isin(BASELINE_CATEGORIES)].copy()
    network_df  = per_ticker[~per_ticker["Category"].isin(BASELINE_CATEGORIES)].copy()

    if baseline_df.empty or network_df.empty:
        print("\n  (Wilcoxon test skipped: need both baseline and network results.)")
        return

    mean_col = f"mean_{rmse_col}"
    baseline_best = (
        baseline_df.groupby(["Category", "Model"])
        .agg(**{mean_col: (rmse_col, "mean")})
        .sort_values(mean_col, ascending=True)
        .reset_index()
        .iloc[0]
    )
    network_best = (
        network_df.groupby(["Category", "Model"])
        .agg(**{mean_col: (rmse_col, "mean")})
        .sort_values(mean_col, ascending=True)
        .reset_index()
        .iloc[0]
    )

    baseline_pair = baseline_df[
        (baseline_df["Category"] == baseline_best["Category"])
        & (baseline_df["Model"] == baseline_best["Model"])
    ][["Ticker", rmse_col]].rename(columns={rmse_col: "baseline"})
    network_pair = network_df[
        (network_df["Category"] == network_best["Category"])
        & (network_df["Model"] == network_best["Model"])
    ][["Ticker", rmse_col]].rename(columns={rmse_col: "network"})

    paired = baseline_pair.merge(network_pair, on="Ticker", how="inner")
    if paired.empty:
        print("\n  (Wilcoxon test skipped: no shared tickers between selected models.)")
        return

    diff    = paired["network"] - paired["baseline"]
    nonzero = paired[diff != 0].copy()
    if nonzero.empty:
        print("\n  (Wilcoxon test skipped: paired differences are all zero.)")
        return

    diff_nz   = nonzero["network"] - nonzero["baseline"]
    test_two  = wilcoxon(nonzero["network"].values, nonzero["baseline"].values, alternative="two-sided")
    test_less = wilcoxon(nonzero["network"].values, nonzero["baseline"].values, alternative="less")

    n_net_wins  = int((diff_nz < 0).sum())
    n_base_wins = int((diff_nz > 0).sum())

    print(f"\n{'=' * 80}")
    print(f"  Wilcoxon Signed-Rank Test: Best Network vs Best Baseline ({metric_label})")
    print(f"{'=' * 80}")
    print(f"  Best baseline : {baseline_best['Category']} / {baseline_best['Model']}  "
          f"(mean {rmse_col} = {baseline_best[mean_col]:.6f})")
    print(f"  Best network  : {network_best['Category']} / {network_best['Model']}  "
          f"(mean {rmse_col} = {network_best[mean_col]:.6f})")
    print(f"  Paired tickers (non-zero diff): {len(nonzero)}")
    print(f"  Network wins / Baseline wins: {n_net_wins} / {n_base_wins}")
    print(f"  Median(network - baseline {rmse_col}): {float(diff_nz.median()):.6f}")
    print(f"  Mean  (network - baseline {rmse_col}): {float(diff_nz.mean()):.6f}")
    print(f"  Two-sided: statistic = {float(test_two.statistic):.4f}, p = {float(test_two.pvalue):.6g}")
    print(f"  One-sided (network < baseline): p = {float(test_less.pvalue):.6g}")


# ---------------------------------------------------------------------------
# Per-ticker network vs baseline
# ---------------------------------------------------------------------------

def print_per_ticker_network_vs_baseline(
    metrics_df: pd.DataFrame,
    tickers: List[str],
    use_log: bool = True,
) -> None:
    """For each selected ticker, show the best network model vs best baseline side by side."""
    rmse_col     = "RMSE_log" if use_log else "RMSE"
    r2_col       = "R2_log"   if use_log else "R2"
    metric_label = "log metrics" if use_log else "raw metrics"

    per_ticker = (
        metrics_df
        .groupby(["Category", "Ticker", "Model"])
        .agg(**{rmse_col: (rmse_col, "mean"), r2_col: (r2_col, "mean")})
        .reset_index()
    )

    baseline_pt = per_ticker[per_ticker["Category"].isin(BASELINE_CATEGORIES)]
    network_pt  = per_ticker[~per_ticker["Category"].isin(BASELINE_CATEGORIES)]
    if baseline_pt.empty or network_pt.empty:
        print("\n  (Per-ticker comparison skipped: need both baseline and network results.)")
        return

    def _best(df: pd.DataFrame):
        means = df.groupby(["Category", "Model"])[rmse_col].mean().reset_index()
        best  = means.loc[means[rmse_col].idxmin()]
        return best["Category"], best["Model"]

    best_base_cat, best_base_model = _best(baseline_pt)
    best_net_cat,  best_net_model  = _best(network_pt)

    print(f"\n{'=' * 80}")
    print(f"  Per-Ticker: Best Network vs Best Baseline  ({metric_label})")
    print(f"{'=' * 80}")
    print(f"  Baseline : {best_base_cat} / {best_base_model}")
    print(f"  Network  : {best_net_cat} / {best_net_model}")
    print()

    rows = []
    for ticker in tickers:
        base_sub = per_ticker[
            (per_ticker["Ticker"] == ticker)
            & (per_ticker["Category"] == best_base_cat)
            & (per_ticker["Model"] == best_base_model)
        ]
        net_sub = per_ticker[
            (per_ticker["Ticker"] == ticker)
            & (per_ticker["Category"] == best_net_cat)
            & (per_ticker["Model"] == best_net_model)
        ]
        if base_sub.empty or net_sub.empty:
            continue
        base_v     = float(base_sub[rmse_col].iloc[0])
        net_v      = float(net_sub[rmse_col].iloc[0])
        base_r2    = float(base_sub[r2_col].iloc[0])
        net_r2     = float(net_sub[r2_col].iloc[0])
        delta_rmse = net_v - base_v
        rows.append({
            "Ticker":          ticker,
            f"Base_{rmse_col}": base_v,
            f"Net_{rmse_col}":  net_v,
            f"Base_{r2_col}":  base_r2,
            f"Net_{r2_col}":   net_r2,
            "Delta_RMSE":      delta_rmse,
            "Winner":          "Network" if delta_rmse < 0 else "Baseline",
        })

    if not rows:
        print("  (No shared tickers found.)")
        return

    df_out = pd.DataFrame(rows).set_index("Ticker")
    with pd.option_context(
        "display.float_format", "{:.6f}".format,
        "display.max_columns", None,
        "display.width", 160,
    ):
        print(df_out.to_string())
    net_wins = sum(1 for r in rows if r["Winner"] == "Network")
    print(f"\n  Network wins {net_wins}/{len(rows)} selected tickers.")


# ---------------------------------------------------------------------------
# k breakdown
# ---------------------------------------------------------------------------

def print_k_breakdown(
    metrics_df_raw: pd.DataFrame,
    use_log: bool = True,
) -> None:
    """Show, for each super-category with k-variants, the best model's key metrics
    at every available k value in side-by-side columns.

    Uses the pre-coalesce (raw/clipped) metrics DataFrame so all k values are present.
    """
    rmse_col     = "RMSE_log" if use_log else "RMSE"
    r2_col       = "R2_log"   if use_log else "R2"
    metric_label = "log metrics" if use_log else "raw metrics"

    k_mask = metrics_df_raw["Category"].str.contains(r"\[k=\d+\]", regex=True)
    if not k_mask.any():
        print("\n  (k-breakdown skipped: no k-variant categories found.)")
        return

    df = metrics_df_raw[k_mask].copy()
    df["super_cat"] = df["Category"].str.replace(r"\s*\[k=\d+\]$", "", regex=True)
    df["k"]         = df["Category"].str.extract(r"\[k=(\d+)\]")[0].astype(int)

    per_ticker = (
        df.groupby(["super_cat", "k", "Ticker", "Model"])
        .agg(**{rmse_col: (rmse_col, "mean"), r2_col: (r2_col, "mean")})
        .reset_index()
    )

    per_model = (
        per_ticker.groupby(["super_cat", "k", "Model"])[rmse_col]
        .mean()
        .reset_index()
    )
    best_idx       = per_model.groupby(["super_cat", "k"])[rmse_col].idxmin()
    best_per_cat_k = per_model.loc[best_idx]

    rows = []
    for _, row in best_per_cat_k.iterrows():
        sub = per_ticker[
            (per_ticker["super_cat"] == row["super_cat"])
            & (per_ticker["k"] == row["k"])
            & (per_ticker["Model"] == row["Model"])
        ]
        rows.append({
            "SuperCategory":    row["super_cat"],
            "k":                int(row["k"]),
            f"mean_{rmse_col}": float(sub[rmse_col].mean()),
            f"mean_{r2_col}":   float(sub[r2_col].mean()),
            "pct_pos":          float((sub[r2_col] > 0).mean()),
        })

    if not rows:
        return

    raw_df     = pd.DataFrame(rows)
    k_values = sorted(raw_df["k"].unique())

    pivot_rmse = raw_df.pivot(index="SuperCategory", columns="k", values=f"mean_{rmse_col}")
    pivot_rmse = pivot_rmse.reindex(columns=k_values)
    pivot_rmse.columns = [f"k{k}_{rmse_col}" for k in pivot_rmse.columns]
    pivot_r2   = raw_df.pivot(index="SuperCategory", columns="k", values=f"mean_{r2_col}")
    pivot_r2   = pivot_r2.reindex(columns=k_values)
    pivot_r2.columns   = [f"k{k}_{r2_col}" for k in pivot_r2.columns]
    pivot_pct  = raw_df.pivot(index="SuperCategory", columns="k", values="pct_pos")
    pivot_pct  = pivot_pct.reindex(columns=k_values)
    pivot_pct.columns  = [f"k{k}_%pos" for k in pivot_pct.columns]

    combined   = pd.concat([pivot_rmse, pivot_r2, pivot_pct], axis=1)
    ordered_cols = (
        [f"k{k}_{rmse_col}" for k in k_values]
        + [f"k{k}_{r2_col}" for k in k_values]
        + [f"k{k}_%pos" for k in k_values]
    )
    combined = combined.reindex(columns=ordered_cols)
    first_rmse = [c for c in combined.columns if rmse_col in c]
    if first_rmse:
        combined = combined.sort_values(first_rmse[0], ascending=True, na_position="last")

    k_label = ", ".join(str(k) for k in k_values)

    print(f"\n{'=' * 80}")
    print(f"  k = {k_label} Breakdown  ({metric_label})")
    print(f"{'=' * 80}")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 60,
    ):
        print(combined.to_string())


# ---------------------------------------------------------------------------
# Weighting-scheme (IDW / Exp / Learned) breakdown
# ---------------------------------------------------------------------------

_WEIGHT_PATTERNS = [
    ("Learned", re.compile(r"LearnedWeight|LW\+", re.IGNORECASE)),
    ("Exp",     re.compile(r"ExpKernel",           re.IGNORECASE)),
    ("IDW",     re.compile(r"Network|Clustering|Split", re.IGNORECASE)),
]


def _weight_scheme(cat: str) -> str:
    """Classify a super-category into IDW / Exp / Learned (returns 'Other' if unclear)."""
    for scheme, pat in _WEIGHT_PATTERNS:
        if pat.search(cat):
            return scheme
    return "Other"


def _graph_base(cat: str) -> str:
    """Classify a super-category by graph construction type (SqCorr / PCorr / MI)."""
    if cat.startswith("PCorr"):
        return "PCorr"
    if cat.startswith("MI ") or cat == "MI":
        return "MI"
    return "SqCorr"


def print_weighting_scheme_breakdown(
    metrics_df: pd.DataFrame,
    use_log: bool = True,
) -> None:
    """Compare IDW vs Exp-kernel vs Learned-Weight schemes across graph types (SqCorr, PCorr, MI).

    Uses the coalesced metrics (k already selected). For each
    (graph_base, weight_scheme) pair the best model (by lowest mean RMSE)
    is displayed.
    """
    rmse_col     = "RMSE_log" if use_log else "RMSE"
    r2_col       = "R2_log"   if use_log else "R2"
    metric_label = "log metrics" if use_log else "raw metrics"

    net_df = metrics_df[~metrics_df["Category"].isin(BASELINE_CATEGORIES)].copy()
    if net_df.empty:
        return

    net_df["graph_base"]    = net_df["Category"].map(_graph_base)
    net_df["weight_scheme"] = net_df["Category"].map(_weight_scheme)

    per_ticker = (
        net_df.groupby(["graph_base", "weight_scheme", "Ticker", "Model"])
        .agg(**{rmse_col: (rmse_col, "mean"), r2_col: (r2_col, "mean")})
        .reset_index()
    )

    rows = []
    for (gb, ws), grp in per_ticker.groupby(["graph_base", "weight_scheme"]):
        if ws == "Other":
            continue
        per_model  = grp.groupby("Model")[rmse_col].mean()
        best_model = per_model.idxmin()
        best_sub   = grp[grp["Model"] == best_model]
        rows.append({
            "GraphBase":       gb,
            "WeightScheme":    ws,
            "BestModel":       best_model,
            f"mean_{rmse_col}": float(best_sub[rmse_col].mean()),
            f"mean_{r2_col}":   float(best_sub[r2_col].mean()),
            "pct_R2_pos":      float((best_sub[r2_col] > 0).mean()),
        })

    if not rows:
        return

    scheme_df = (
        pd.DataFrame(rows)
        .set_index(["GraphBase", "WeightScheme"])
        .sort_values(f"mean_{rmse_col}", ascending=True)
    )
    print(f"\n{'=' * 80}")
    print(f"  Weighting Scheme: IDW vs Exp vs Learned  ({metric_label})")
    print(f"{'=' * 80}")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 80,
    ):
        print(scheme_df.to_string())


# ---------------------------------------------------------------------------
# Presentation-mode helpers and tables
# ---------------------------------------------------------------------------

def _structure_cat(cat: str) -> str:
    """Feature-structure type: Plain / Clustering / Split / Split+Cluster."""
    if "Split+Clustering" in cat:
        return "Split+Cluster"
    if re.search(r"LW\+|Clustering", cat):
        return "Clustering"
    if re.search(r"SplitFeatures|Split", cat):
        return "Split"
    return "Plain"


def _present_weight_scheme(cat: str) -> str:
    """Weighting scheme — catches 'Exp+' prefix that the tag-based _weight_scheme misses."""
    if re.search(r"LearnedWeight|LW\+", cat, re.IGNORECASE):
        return "Learned"
    if re.search(r"Exp", cat, re.IGNORECASE):
        return "Exp"
    return "IDW"


def _graph_type_label(cat: str) -> str:
    """Human-readable graph distance type for a category string."""
    if cat.startswith("PCorr"):
        return "Partial Corr."
    if cat.startswith("MI ") or cat == "MI":
        return "Mutual Info."
    return "Sq. Corr."


_PRESENT_WEIGHT_READABLE: dict = {
    "Learned": "Learned Weights",
    "Exp":     "Exp. Kernel",
    "IDW":     "IDW",
}
_PRESENT_STRUCT_READABLE: dict = {
    "Plain":         "Plain",
    "Clustering":    "Clustering",
    "Split":         "Split Features",
    "Split+Cluster": "Split + Cluster",
}


def _decode_present_cat(cat: str) -> dict:
    """Return readable labels for the three dimensions of a category string."""
    return {
        "distance":  _graph_type_label(cat),
        "weighting": _PRESENT_WEIGHT_READABLE[_present_weight_scheme(cat)],
        "structure": _PRESENT_STRUCT_READABLE.get(_structure_cat(cat), _structure_cat(cat)),
    }


def _best_model_label(model: str) -> str:
    """Shorten a raw model name for display in presentation tables."""
    m = re.sub(r"\s*\(no outliers\)", "", model).strip()
    for pat, repl in [
        (r"^NetHAR\+CSplit\b",    "HAR+CS"),
        (r"^NetHAR\+C\b",         "HAR+C"),
        (r"^NetHAR-Split\b",      "HAR-Split"),
        (r"^NetHAR\b",            "HAR"),
        (r"^NetworkVAR\+CSplit\b", "VAR+CS"),
        (r"^NetworkVAR\+C\b",     "VAR+C"),
        (r"^NetworkVAR\b",        "VAR"),
        (r"^LearnedW\+C\b",       "LW+C"),
        (r"^LearnedW\b",          "LW"),
    ]:
        m = re.sub(pat, repl, m)
    m = re.sub(r"Lasso a=", "L=", m)
    m = re.sub(r"Ridge a=", "R=", m)
    m = re.sub(r"\ba=([^,)]+),\s*b=", r"α=\1,β=", m)
    return re.sub(r"\s+", " ", m).strip()


def _best_in_group(
    net_pt: pd.DataFrame,
    mask: "pd.Series",
    rmse_col: str,
    r2_col: str,
) -> "dict | None":
    """Pick the best (Category, Model) pair from net_pt[mask] by lowest mean RMSE.
    Returns a stats dict or None if the group is empty.
    """
    grp = net_pt[mask]
    if grp.empty:
        return None
    means = grp.groupby(["Category", "Model"])[rmse_col].mean().reset_index()
    best  = means.loc[means[rmse_col].idxmin()]
    sub   = grp[
        (grp["Category"] == best["Category"]) & (grp["Model"] == best["Model"])
    ]
    return {
        "Category":   best["Category"],
        "Model":      best["Model"],
        "mean_rmse":  float(sub[rmse_col].mean()),
        "mean_r2":    float(sub[r2_col].mean()),
        "pct_r2_pos": float((sub[r2_col] > 0).mean()),
    }


def _present_per_ticker_table(
    per_ticker: pd.DataFrame,
    tickers: list,
    base_cat: str,
    base_model: str,
    net_cat: str,
    net_model: str,
    rmse_col: str,
    r2_col: str,
    title: str,
) -> None:
    rows = []
    for ticker in tickers:
        base_row = per_ticker[
            (per_ticker["Ticker"] == ticker)
            & (per_ticker["Category"] == base_cat)
            & (per_ticker["Model"] == base_model)
        ]
        net_row = per_ticker[
            (per_ticker["Ticker"] == ticker)
            & (per_ticker["Category"] == net_cat)
            & (per_ticker["Model"] == net_model)
        ]
        if base_row.empty or net_row.empty:
            continue
        b_rmse = float(base_row[rmse_col].iloc[0])
        n_rmse = float(net_row[rmse_col].iloc[0])
        b_r2   = float(base_row[r2_col].iloc[0])
        n_r2   = float(net_row[r2_col].iloc[0])
        rows.append({
            "Ticker":           ticker,
            f"Base {rmse_col}": b_rmse,
            f"Net  {rmse_col}": n_rmse,
            f"Base {r2_col}":   b_r2,
            f"Net  {r2_col}":   n_r2,
        })
    if not rows:
        return
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    df_out = pd.DataFrame(rows).set_index("Ticker")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_columns", None,
        "display.width", 200,
    ):
        print(df_out.to_string())
    net_wins = sum(1 for r in rows if r[f"Net  {rmse_col}"] < r[f"Base {rmse_col}"])
    print(f"  → Network wins {net_wins}/{len(rows)} tickers shown.")


def _print_structural_table(
    net_pt: pd.DataFrame,
    group_col: str,
    group_defs: list,
    primary_col: str,
    extra_decode,
    rmse_col: str,
    r2_col: str,
    title: str,
) -> None:
    """Print one structural-comparison table.

    Parameters
    ----------
    group_col   : Column in net_pt holding the grouping value (e.g. '_dist').
    group_defs  : List of (display_label, filter_value) tuples.
    primary_col : Left-most index column name in the output.
    extra_decode: Callable(cat: str) → dict of extra column name → value.
    """
    rows = []
    for display, fval in group_defs:
        result = _best_in_group(net_pt, net_pt[group_col] == fval, rmse_col, r2_col)
        if result is None:
            continue
        extra = extra_decode(result["Category"])
        rows.append({
            primary_col:        display,
            **extra,
            "Best Model":       _best_model_label(result["Model"]),
            f"mean {rmse_col}": result["mean_rmse"],
            f"mean {r2_col}":   result["mean_r2"],
            "%R²>0":            f"{result['pct_r2_pos']:.1%}",
        })
    if not rows:
        return
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    df = pd.DataFrame(rows).set_index(primary_col)
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 40,
    ):
        print(df.to_string())


def print_presentation_tables(
    metrics_df: pd.DataFrame,
    use_log: bool = True,
    selection: str = "r2",
    named_tickers: "Optional[List[str]]" = None,
) -> None:
    """Generate presentation-ready summary tables.

    1. Per-ticker comparison (best network vs best baseline):
       - Top 5 tickers with the largest network improvement.
       - Bottom 5 tickers with the smallest improvement (or regression).
       - Named tickers: AAPL, TSLA, GOOG, META, MSFT, NVDA, NFLX, AMZN.
    2. Wilcoxon signed-rank test.
    3. Three structural comparison tables (one best-model row per group):
       - Graph distance:    Sq. Corr. / Partial Corr. / Mutual Info.
       - Weighting scheme:  IDW / Exp. Kernel / Learned Weights.
       - Feature structure: Plain / Clustering / Split Features / Split+Cluster.

    Only log metrics are shown when *use_log* is True; only raw metrics otherwise.
    """
    rmse_col     = "RMSE_log" if use_log else "RMSE"
    r2_col       = "R2_log"   if use_log else "R2"
    metric_label = "log" if use_log else "raw"

    per_ticker = (
        metrics_df
        .groupby(["Category", "Ticker", "Model"])
        .agg(**{rmse_col: (rmse_col, "mean"), r2_col: (r2_col, "mean")})
        .reset_index()
    )

    baseline_pt = per_ticker[per_ticker["Category"].isin(BASELINE_CATEGORIES)]
    network_pt  = per_ticker[~per_ticker["Category"].isin(BASELINE_CATEGORIES)].copy()

    if baseline_pt.empty or network_pt.empty:
        print("\n  (Presentation tables skipped: need both baseline and network results.)")
        return

    def _pick_best(df: pd.DataFrame):
        means = df.groupby(["Category", "Model"])[rmse_col].mean().reset_index()
        row   = means.loc[means[rmse_col].idxmin()]
        return row["Category"], row["Model"]

    best_base_cat, best_base_model = _pick_best(baseline_pt)
    best_net_cat,  best_net_model  = _pick_best(network_pt)

    print(f"\n{'=' * 80}")
    print(f"  PRESENTATION TABLES  ({metric_label} metrics)")
    print(f"{'=' * 80}")
    print(f"  Best baseline : {best_base_cat} / {best_base_model}")
    print(f"  Best network  : {best_net_cat}  / {best_net_model}")

    # --- Per-ticker delta: find top-5 and bottom-5 ---
    base_ser = baseline_pt[
        (baseline_pt["Category"] == best_base_cat)
        & (baseline_pt["Model"] == best_base_model)
    ][["Ticker", rmse_col]].rename(columns={rmse_col: "__base"})
    net_ser = network_pt[
        (network_pt["Category"] == best_net_cat)
        & (network_pt["Model"] == best_net_model)
    ][["Ticker", rmse_col]].rename(columns={rmse_col: "__net"})
    delta_df = base_ser.merge(net_ser, on="Ticker", how="inner").copy()
    delta_df["__delta"] = delta_df["__net"] - delta_df["__base"]
    delta_df = delta_df.sort_values("__delta")
    top5    = delta_df.head(5)["Ticker"].tolist()
    bottom5 = delta_df.tail(5)["Ticker"].tolist()

    # --- Per-ticker tables ---
    print(f"\n{'=' * 80}")
    print(f"  Per-Ticker Tables")
    print(f"  Baseline : {best_base_cat} / {_best_model_label(best_base_model)}")
    print(f"  Network  : {best_net_cat}  / {_best_model_label(best_net_model)}")
    print(f"{'=' * 80}")
    _present_per_ticker_table(
        per_ticker, top5,
        best_base_cat, best_base_model, best_net_cat, best_net_model,
        rmse_col, r2_col, "Top 5: Largest Network Improvement",
    )
    _present_per_ticker_table(
        per_ticker, bottom5,
        best_base_cat, best_base_model, best_net_cat, best_net_model,
        rmse_col, r2_col, "Bottom 5: Smallest Network Improvement",
    )
    if named_tickers is None:
        named_tickers = ["AAPL", "TSLA", "GOOG", "META", "MSFT", "NVDA", "NFLX", "AMZN"]
    _present_per_ticker_table(
        per_ticker,
        named_tickers,
        best_base_cat, best_base_model, best_net_cat, best_net_model,
        rmse_col, r2_col, "Named Tickers",
    )

    # --- Wilcoxon test ---
    print_wilcoxon_best_network_vs_baseline(metrics_df, use_log=use_log)

    # --- Structural breakdown tables ---
    network_pt["_dist"]   = network_pt["Category"].map(_graph_type_label)
    network_pt["_weight"] = network_pt["Category"].map(_present_weight_scheme)
    network_pt["_struct"] = network_pt["Category"].map(_structure_cat)

    _print_structural_table(
        network_pt,
        group_col="_dist",
        group_defs=[
            ("Sq. Corr.",     "Sq. Corr."),
            ("Partial Corr.", "Partial Corr."),
            ("Mutual Info.",  "Mutual Info."),
        ],
        primary_col="Distance",
        extra_decode=lambda cat: {
            "Weighting": _decode_present_cat(cat)["weighting"],
            "Structure": _decode_present_cat(cat)["structure"],
        },
        rmse_col=rmse_col,
        r2_col=r2_col,
        title=f"Graph Distance Comparison  ({metric_label} metrics)",
    )
    _print_structural_table(
        network_pt,
        group_col="_weight",
        group_defs=[
            ("IDW",             "IDW"),
            ("Exp. Kernel",     "Exp"),
            ("Learned Weights", "Learned"),
        ],
        primary_col="Weighting",
        extra_decode=lambda cat: {
            "Distance":  _decode_present_cat(cat)["distance"],
            "Structure": _decode_present_cat(cat)["structure"],
        },
        rmse_col=rmse_col,
        r2_col=r2_col,
        title=f"Weighting Scheme Comparison  ({metric_label} metrics)",
    )
    _print_structural_table(
        network_pt,
        group_col="_struct",
        group_defs=[
            ("Plain",          "Plain"),
            ("Clustering",     "Clustering"),
            ("Split Features", "Split"),
            ("Split + Cluster","Split+Cluster"),
        ],
        primary_col="Structure",
        extra_decode=lambda cat: {
            "Distance":  _decode_present_cat(cat)["distance"],
            "Weighting": _decode_present_cat(cat)["weighting"],
        },
        rmse_col=rmse_col,
        r2_col=r2_col,
        title=f"Feature Structure Comparison  ({metric_label} metrics)",
    )


def print_presentation_report(
    metrics_df_raw: pd.DataFrame,
    metrics_df: pd.DataFrame,
    use_log: bool = True,
    named_tickers: "Optional[List[str]]" = None,
) -> None:
    """Print the concise regression report used for presentation logs."""
    print_k_breakdown(metrics_df_raw, use_log=use_log)
    print_weighting_scheme_breakdown(metrics_df, use_log=use_log)
    print_presentation_tables(
        metrics_df,
        use_log=use_log,
        named_tickers=named_tickers,
    )


# ---------------------------------------------------------------------------
# Load-from-disk entry point
# ---------------------------------------------------------------------------

def load_and_print_results(
    results_dir: "str | Path" = None,
    sample_tickers: "Optional[List[str]]" = None,
    r2_threshold: float = -1.0,
    plot: bool = False,
    use_log: bool = True,
    selection: str = "r2",
    present: bool = False,
    present_only: bool = False,
    raw_metric_display_scale: float = 1.0,
    raw_metric_display_label: str | None = None,
) -> None:
    """
    Load saved JSON results and reprint all summary tables.

    Categories with ``[k=N]`` suffixes are coalesced into super-categories
    (e.g. ``'Network [k=1/3/5]'`` → ``'Network'``) by selecting the best
    ``(k, model)`` pair per super-category.

    Parameters
    ----------
    results_dir    : Folder containing ``results_bench.json``.
                     Defaults to ``<repo_root>/FinanceNetworks/results``.
    sample_tickers : Tickers for per-ticker tables.
                     Defaults to ``["AAPL", "TSLA", "GOOG", "META", "MSFT"]``.
    r2_threshold   : Kept for API compatibility; no longer used.
    plot           : If True, regenerate and save the summary bar chart.
    use_log        : When True (default), log-scale metrics (RMSE_log, R2_log)
                     are used as the primary ranking criterion.
                     Set to False to rank by raw RMSE/R2.
    selection      : Model selection criterion. One of:
                     ``'r2'`` (default) – rank by pct_R2_pos;
                     ``'mean_rmse'``    – rank by mean RMSE (lower is better);
                     ``'median_rmse'``  – rank by median RMSE (lower is better).
    present        : If True, generate condensed presentation-ready tables instead of
                     (or in addition to) the full diagnostic output.
    raw_metric_display_scale : Display-only multiplier applied to raw RMSE / MAE
                     columns after loading results. Useful when comparing data
                     sources stored in different but equivalent variance units.
    raw_metric_display_label : Optional explanatory label printed when
                     ``raw_metric_display_scale`` is not 1.0.
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

    metric_label = "log metrics" if use_log else "raw metrics"
    print(f"Loading fold-level metrics from {raw_path} ...")
    print(f"Priority metric: {metric_label}")

    # Step 1: load and drop numerical blowups at fold level
    metrics_df_raw = pd.read_json(raw_path, orient="records")
    metrics_df_raw = clip_extreme_metrics(metrics_df_raw)
    metrics_df_raw = _scale_metric_columns_for_display(
        metrics_df_raw,
        raw_metric_scale=raw_metric_display_scale,
    )

    if raw_metric_display_scale != 1.0:
        label = raw_metric_display_label or f"x{raw_metric_display_scale:g}"
        print(
            "Display scaling: raw RMSE / MAE columns are multiplied by "
            f"{label}."
        )

    # Step 2: coalesce k-variants into super-categories
    metrics_df = coalesce_categories(metrics_df_raw)

    # Step 3: build summary (also filters degenerate models)
    summary = summarize_benchmarks(metrics_df, use_log=use_log, selection=selection)

    if present_only:
        print_presentation_report(
            metrics_df_raw,
            metrics_df,
            use_log=use_log,
            named_tickers=sample_tickers,
        )
    else:
        print_summary(summary, title=f"Full Summary ({metric_label})")
        print_best_per_category(summary, use_log=use_log, selection=selection)
        print_compact_leaderboard(summary, use_log=use_log, selection=selection)

        # Network vs baseline comparisons
        print_wilcoxon_best_network_vs_baseline(metrics_df, use_log=use_log)

        # Per-ticker breakdowns
        print_per_ticker_network_vs_baseline(metrics_df, sample_tickers, use_log=use_log)
        print_per_ticker_tables(metrics_df, sample_tickers, use_log=use_log, selection=selection)

        # Structural breakdowns (k-breakdown uses pre-coalesce data)
        print_k_breakdown(metrics_df_raw, use_log=use_log)
        print_weighting_scheme_breakdown(metrics_df, use_log=use_log)

        if present:
            print_presentation_tables(metrics_df, use_log=use_log, selection=selection,
                                      named_tickers=sample_tickers)

    print(f"\n  (Loaded {len(metrics_df_raw)} rows from {raw_path})")

    if plot:
        from visualize.plot_model_results import plot_summary_metrics
        plot_summary_metrics(
            summary,
            save_path=str(results_dir / "summary_metrics.png"),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print regression benchmark results from saved JSON.",
    )
    parser.add_argument(
        "--metric",
        choices=["log", "raw"],
        default="log",
        help=(
            "Priority metric for ranking and selecting 'best' models. "
            "'log' (default) uses RMSE_log / R2_log; "
            "'raw' uses RMSE / R2."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=["r2", "mean_rmse", "median_rmse"],
        default="r2",
        help=(
            "Model selection criterion. "
            "'r2' (default) ranks by pct R2 > 0; "
            "'mean_rmse' ranks by mean RMSE (lower is better); "
            "'median_rmse' ranks by median RMSE (lower is better)."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "TSLA", "GOOG", "META", "MSFT"],
        metavar="TICKER",
        help="Sample tickers for per-ticker tables (default: AAPL TSLA GOOG META MSFT).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help="Path to results directory (default: <repo>/FinanceNetworks/results).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate and save the summary bar chart.",
    )
    parser.add_argument(
        "--present",
        action="store_true",
        help=(
            "Generate condensed presentation-ready tables: per-ticker top/bottom-5 "
            "improvement, named-ticker table, Wilcoxon test, and three structural "
            "comparison tables (distance / weighting / structure)."
        ),
    )
    parser.add_argument(
        "--present-only",
        action="store_true",
        help=(
            "Print only the presentation-oriented regression report: k breakdown, "
            "weighting-scheme breakdown, Wilcoxon, and all presentation tables."
        ),
    )
    args = parser.parse_args()
    load_and_print_results(
        results_dir=args.results_dir,
        sample_tickers=args.tickers,
        plot=args.plot,
        use_log=(args.metric == "log"),
        selection=args.selection,
        present=args.present,
        present_only=args.present_only,
    )
