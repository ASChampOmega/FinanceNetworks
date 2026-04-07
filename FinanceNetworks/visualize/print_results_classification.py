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
print_presentation_report_clf     : Print the concise presentation-only report.
load_and_print_classification_results : Load saved JSON results and reprint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from visualize.utils import coalesce_categories

# Re-use structural helpers from the regression printing module so that
# both regression and classification presentation tables share the same
# graph-distance / weighting / structure labelling logic.
from visualize.print_results import (
    _weight_scheme,
    _graph_base,
    _graph_type_label,
    _structure_cat,
    _present_weight_scheme,
    _PRESENT_WEIGHT_READABLE,
    _PRESENT_STRUCT_READABLE,
    _decode_present_cat,
    _best_model_label,
    _WEIGHT_PATTERNS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_CATEGORIES_CLF: frozenset = frozenset({
    "HAR-Logit", "HAR-Ext-Logit",
    "RegimeSwitching-Logit", "DCC-GARCH-Logit",
})


# ---------------------------------------------------------------------------
# Selection-metric config
# ---------------------------------------------------------------------------

_SEL_MAP: dict = {
    # selection_key: (summary_col, ascending)
    "roc_auc":         ("mean_ROC_AUC",         False),
    "f1":              ("mean_F1",               False),
    "weighted_recall": ("mean_Weighted_Recall",  False),
}

_TICKER_SEL_MAP: dict = {
    # selection_key: per-ticker column name
    "roc_auc":         "ROC_AUC",
    "f1":              "F1",
    "weighted_recall": "Weighted_Recall",
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_classification(
    metrics_df: pd.DataFrame,
    selection: str = "roc_auc",
) -> pd.DataFrame:
    """
    Aggregate fold-level classification metrics.

    Two-stage aggregation:
      1. Average across folds per (Category, Ticker, Model).
      2. Summarise across tickers per (Category, Model).

    Parameters
    ----------
    selection : str
        Metric used for sorting.  One of ``'roc_auc'`` (default),
        ``'f1'``, ``'weighted_recall'``.

    Returns a MultiIndex DataFrame sorted by the chosen metric descending.
    """
    sort_col, sort_asc = _SEL_MAP.get(selection, _SEL_MAP["roc_auc"])

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
        .sort_values(sort_col, ascending=sort_asc)
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
    "display.width": 220,
    "display.max_colwidth": 60,
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


def print_best_classifier(
    summary: pd.DataFrame,
    selection: str = "roc_auc",
) -> None:
    """
    Print a compact table showing the single best classifier per category.
    """
    sel_col, _ = _SEL_MAP.get(selection, _SEL_MAP["roc_auc"])
    print(f"\n{'=' * 80}")
    print(f"  Best Classifier per Category (by {sel_col})")
    print(f"{'=' * 80}")
    best_rows = []
    for cat in summary.index.get_level_values("Category").unique():
        cat_df = summary.loc[cat]
        best_model = cat_df[sel_col].idxmax()
        row = cat_df.loc[best_model]
        best_rows.append({"Category": cat, "Model": best_model, **row.to_dict()})
    best_df = pd.DataFrame(best_rows).set_index(["Category", "Model"])
    cols = [c for c in _PRINT_COLS if c in best_df.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        print(best_df[cols].sort_values(sel_col, ascending=False).to_string())


def print_overall_best(
    summary: pd.DataFrame,
    selection: str = "roc_auc",
) -> None:
    """
    Print the single globally best model across all categories.
    """
    sel_col, _ = _SEL_MAP.get(selection, _SEL_MAP["roc_auc"])
    print(f"\n{'=' * 80}")
    print(f"  Overall Best Classifier (by {sel_col} across all tickers)")
    print(f"{'=' * 80}")
    best_idx = summary[sel_col].idxmax()
    best_row = summary.loc[[best_idx]]
    cols = [c for c in _PRINT_COLS if c in best_row.columns]
    with pd.option_context(*sum(_FMT.items(), ())):
        print(best_row[cols].to_string())


def print_per_ticker_classification(
    metrics_df: pd.DataFrame,
    tickers: List[str],
    selection: str = "roc_auc",
) -> None:
    """
    Print per-ticker classification metrics for the best model per category.
    """
    sel_metric = _TICKER_SEL_MAP.get(selection, "ROC_AUC")

    # Identify the best model per category by cross-ticker mean of sel_metric
    per_ticker = (
        metrics_df.groupby(["Category", "Ticker", "Model"])
        .agg(**{sel_metric: (sel_metric, "mean")})
        .reset_index()
    )
    mean_sel = (
        per_ticker.groupby(["Category", "Model"])
        .agg(**{f"mean_{sel_metric}": (sel_metric, "mean")})
        .reset_index()
    )
    best_models: dict = {}
    for cat, grp in mean_sel.groupby("Category"):
        best_models[cat] = grp.loc[grp[f"mean_{sel_metric}"].idxmax(), "Model"]

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
            if sel_metric in tdf.columns and tdf[sel_metric].notna().any():
                best_idx = tdf[sel_metric].idxmax()
                best_score = tdf[sel_metric].max()
                try:
                    best_category, best_model = best_idx
                except Exception:
                    best_category, best_model = (None, best_idx)
                print(f"\nBest Model for {ticker}: {best_model} "
                      f"(Category: {best_category}, {sel_metric}: {best_score:.4f})")
            elif sel_metric in tdf.columns:
                print(f"\n  (All {sel_metric} values are NaN for {ticker})")


def print_compact_clf_leaderboard(
    summary: pd.DataFrame,
    selection: str = "roc_auc",
) -> None:
    """Ultra-short best-per-category table with only the key metrics."""
    sel_col, _ = _SEL_MAP.get(selection, _SEL_MAP["roc_auc"])
    print(f"\n{'=' * 80}")
    print(f"  Leaderboard (best classifier per category, selection={selection})")
    print(f"{'=' * 80}")
    rows = []
    for cat in summary.index.get_level_values("Category").unique():
        cat_df = summary.loc[cat]
        best_model = cat_df[sel_col].idxmax()
        r = cat_df.loc[best_model]
        rows.append({
            "Category": cat,
            "Model": best_model,
            "ROC_AUC": r.get("mean_ROC_AUC", float("nan")),
            "F1": r.get("mean_F1", float("nan")),
            "W.Recall": r.get("mean_Weighted_Recall", float("nan")),
            "Prec": r.get("mean_Precision", float("nan")),
        })
    lb = pd.DataFrame(rows).set_index(["Category", "Model"])
    lb = lb.sort_values("ROC_AUC", ascending=False)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 220, "display.max_colwidth", 60):
        print(lb.to_string())


# ---------------------------------------------------------------------------
# Per-ticker: network vs baseline
# ---------------------------------------------------------------------------

def print_per_ticker_network_vs_baseline_clf(
    metrics_df: pd.DataFrame,
    tickers: List[str],
    selection: str = "roc_auc",
) -> None:
    """For each selected ticker, show the best network classifier vs best baseline side by side."""
    sel_metric = _TICKER_SEL_MAP.get(selection, "ROC_AUC")

    per_ticker = (
        metrics_df
        .groupby(["Category", "Ticker", "Model"])
        .agg(**{sel_metric: (sel_metric, "mean"), "F1": ("F1", "mean")})
        .reset_index()
    )

    baseline_pt = per_ticker[per_ticker["Category"].isin(BASELINE_CATEGORIES_CLF)]
    network_pt  = per_ticker[~per_ticker["Category"].isin(BASELINE_CATEGORIES_CLF)]

    if baseline_pt.empty or network_pt.empty:
        print("\n  (Per-ticker comparison skipped: need both baseline and network results.)")
        return

    def _best(df: pd.DataFrame):
        means = df.groupby(["Category", "Model"])[sel_metric].mean().reset_index()
        best  = means.loc[means[sel_metric].idxmax()]
        return best["Category"], best["Model"]

    best_base_cat, best_base_model = _best(baseline_pt)
    best_net_cat,  best_net_model  = _best(network_pt)

    print(f"\n{'=' * 80}")
    print(f"  Per-Ticker: Best Network vs Best Baseline  (by {sel_metric})")
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
        base_v  = float(base_sub[sel_metric].iloc[0])
        net_v   = float(net_sub[sel_metric].iloc[0])
        base_f1 = float(base_sub["F1"].iloc[0])
        net_f1  = float(net_sub["F1"].iloc[0])
        delta   = net_v - base_v
        rows.append({
            "Ticker":              ticker,
            f"Base_{sel_metric}":  base_v,
            f"Net_{sel_metric}":   net_v,
            "Base_F1":             base_f1,
            "Net_F1":              net_f1,
            f"Delta_{sel_metric}": delta,
            "Winner":              "Network" if delta > 0 else "Baseline",
        })

    if not rows:
        print("  (No shared tickers found.)")
        return

    df_out = pd.DataFrame(rows).set_index("Ticker")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_columns", None,
        "display.width", 160,
    ):
        print(df_out.to_string())
    net_wins = sum(1 for r in rows if r["Winner"] == "Network")
    print(f"\n  Network wins {net_wins}/{len(rows)} selected tickers.")


# ---------------------------------------------------------------------------
# k breakdown (classification)
# ---------------------------------------------------------------------------

def print_k_breakdown_clf(
    metrics_df_raw: pd.DataFrame,
    selection: str = "roc_auc",
) -> None:
    """Show per-k breakdown of the best classifier for each super-category."""
    sel_metric = _TICKER_SEL_MAP.get(selection, "ROC_AUC")

    k_mask = metrics_df_raw["Category"].str.contains(r"\[k=\d+\]", regex=True)
    if not k_mask.any():
        print("\n  (k-breakdown skipped: no k-variant categories found.)")
        return

    df = metrics_df_raw[k_mask].copy()
    df["super_cat"] = df["Category"].str.replace(r"\s*\[k=\d+\]$", "", regex=True)
    df["k"]         = df["Category"].str.extract(r"\[k=(\d+)\]")[0].astype(int)

    per_ticker = (
        df.groupby(["super_cat", "k", "Ticker", "Model"])
        .agg(**{sel_metric: (sel_metric, "mean"), "F1": ("F1", "mean")})
        .reset_index()
    )

    per_model = (
        per_ticker.groupby(["super_cat", "k", "Model"])[sel_metric]
        .mean()
        .reset_index()
    )
    best_idx       = per_model.groupby(["super_cat", "k"])[sel_metric].idxmax()
    best_per_cat_k = per_model.loc[best_idx]

    rows = []
    for _, row in best_per_cat_k.iterrows():
        sub = per_ticker[
            (per_ticker["super_cat"] == row["super_cat"])
            & (per_ticker["k"] == row["k"])
            & (per_ticker["Model"] == row["Model"])
        ]
        rows.append({
            "SuperCategory":       row["super_cat"],
            "k":                   int(row["k"]),
            f"mean_{sel_metric}":  float(sub[sel_metric].mean()),
            "mean_F1":             float(sub["F1"].mean()),
        })

    if not rows:
        return

    raw_df    = pd.DataFrame(rows)
    k_values = sorted(raw_df["k"].unique())

    pivot_sel = raw_df.pivot(index="SuperCategory", columns="k", values=f"mean_{sel_metric}")
    pivot_sel = pivot_sel.reindex(columns=k_values)
    pivot_sel.columns = [f"k{k}_{sel_metric}" for k in pivot_sel.columns]
    pivot_f1  = raw_df.pivot(index="SuperCategory", columns="k", values="mean_F1")
    pivot_f1  = pivot_f1.reindex(columns=k_values)
    pivot_f1.columns  = [f"k{k}_F1" for k in pivot_f1.columns]

    combined = pd.concat([pivot_sel, pivot_f1], axis=1)
    ordered_cols = [f"k{k}_{sel_metric}" for k in k_values] + [f"k{k}_F1" for k in k_values]
    combined = combined.reindex(columns=ordered_cols)
    first_sel = [c for c in combined.columns if sel_metric in c]
    if first_sel:
        combined = combined.sort_values(first_sel[0], ascending=False, na_position="last")

    k_label = ", ".join(str(k) for k in k_values)

    print(f"\n{'=' * 80}")
    print(f"  k = {k_label} Breakdown  (by {sel_metric})")
    print(f"{'=' * 80}")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 60,
    ):
        print(combined.to_string())


# ---------------------------------------------------------------------------
# Weighting-scheme breakdown (classification)
# ---------------------------------------------------------------------------

def print_weighting_scheme_breakdown_clf(
    metrics_df: pd.DataFrame,
    selection: str = "roc_auc",
) -> None:
    """Compare IDW vs Exp-kernel vs Learned-Weight schemes across graph types."""
    sel_metric = _TICKER_SEL_MAP.get(selection, "ROC_AUC")

    net_df = metrics_df[~metrics_df["Category"].isin(BASELINE_CATEGORIES_CLF)].copy()
    if net_df.empty:
        return

    net_df["graph_base"]    = net_df["Category"].map(_graph_base)
    net_df["weight_scheme"] = net_df["Category"].map(_weight_scheme)

    per_ticker = (
        net_df.groupby(["graph_base", "weight_scheme", "Ticker", "Model"])
        .agg(**{sel_metric: (sel_metric, "mean"), "F1": ("F1", "mean")})
        .reset_index()
    )

    rows = []
    for (gb, ws), grp in per_ticker.groupby(["graph_base", "weight_scheme"]):
        if ws == "Other":
            continue
        per_model  = grp.groupby("Model")[sel_metric].mean()
        best_model = per_model.idxmax()
        best_sub   = grp[grp["Model"] == best_model]
        rows.append({
            "GraphBase":          gb,
            "WeightScheme":       ws,
            "BestModel":          best_model,
            f"mean_{sel_metric}": float(best_sub[sel_metric].mean()),
            "mean_F1":            float(best_sub["F1"].mean()),
        })

    if not rows:
        return

    scheme_df = (
        pd.DataFrame(rows)
        .set_index(["GraphBase", "WeightScheme"])
        .sort_values(f"mean_{sel_metric}", ascending=False)
    )
    print(f"\n{'=' * 80}")
    print(f"  Weighting Scheme: IDW vs Exp vs Learned  (by {sel_metric})")
    print(f"{'=' * 80}")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 80,
    ):
        print(scheme_df.to_string())


# ---------------------------------------------------------------------------
# Presentation-mode helpers (classification)
# ---------------------------------------------------------------------------

def _best_in_group_clf(
    net_pt: pd.DataFrame,
    mask: "pd.Series",
    sel_metric: str,
) -> "dict | None":
    """Pick the best (Category, Model) pair from net_pt[mask] by highest mean sel_metric."""
    grp = net_pt[mask]
    if grp.empty:
        return None
    means = grp.groupby(["Category", "Model"])[sel_metric].mean().reset_index()
    best  = means.loc[means[sel_metric].idxmax()]
    sub   = grp[
        (grp["Category"] == best["Category"]) & (grp["Model"] == best["Model"])
    ]
    return {
        "Category":    best["Category"],
        "Model":       best["Model"],
        "mean_metric": float(sub[sel_metric].mean()),
        "mean_f1":     float(sub["F1"].mean()),
    }


def _present_per_ticker_table_clf(
    per_ticker: pd.DataFrame,
    tickers: list,
    base_cat: str,
    base_model: str,
    net_cat: str,
    net_model: str,
    sel_metric: str,
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
        b_sel = float(base_row[sel_metric].iloc[0])
        n_sel = float(net_row[sel_metric].iloc[0])
        b_f1  = float(base_row["F1"].iloc[0])
        n_f1  = float(net_row["F1"].iloc[0])
        rows.append({
            "Ticker":              ticker,
            f"Base {sel_metric}":  b_sel,
            f"Net  {sel_metric}":  n_sel,
            "Base F1":             b_f1,
            "Net  F1":             n_f1,
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
    net_wins = sum(1 for r in rows if r[f"Net  {sel_metric}"] > r[f"Base {sel_metric}"])
    print(f"  → Network wins {net_wins}/{len(rows)} tickers shown.")


def _print_structural_table_clf(
    net_pt: pd.DataFrame,
    group_col: str,
    group_defs: list,
    primary_col: str,
    extra_decode,
    sel_metric: str,
    title: str,
) -> None:
    """Print one structural-comparison table for classification."""
    rows = []
    for display, fval in group_defs:
        result = _best_in_group_clf(net_pt, net_pt[group_col] == fval, sel_metric)
        if result is None:
            continue
        extra = extra_decode(result["Category"])
        rows.append({
            primary_col:             display,
            **extra,
            "Best Model":            _best_model_label(result["Model"]),
            f"mean {sel_metric}":    result["mean_metric"],
            "mean F1":               result["mean_f1"],
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


def print_presentation_tables_clf(
    metrics_df: pd.DataFrame,
    selection: str = "roc_auc",
    named_tickers: "Optional[List[str]]" = None,
) -> None:
    """Generate presentation-ready summary tables for classification.

    1. Per-ticker comparison (best network vs best baseline):
       - Top 5, Bottom 5, named tickers.
    2. Wilcoxon signed-rank test.
    3. Structural comparison tables (distance / weighting / structure).
    """
    sel_metric = _TICKER_SEL_MAP.get(selection, "ROC_AUC")

    per_ticker = (
        metrics_df
        .groupby(["Category", "Ticker", "Model"])
        .agg(**{sel_metric: (sel_metric, "mean"), "F1": ("F1", "mean")})
        .reset_index()
    )

    baseline_pt = per_ticker[per_ticker["Category"].isin(BASELINE_CATEGORIES_CLF)]
    network_pt  = per_ticker[~per_ticker["Category"].isin(BASELINE_CATEGORIES_CLF)].copy()

    if baseline_pt.empty or network_pt.empty:
        print("\n  (Presentation tables skipped: need both baseline and network results.)")
        return

    def _pick_best(df: pd.DataFrame):
        means = df.groupby(["Category", "Model"])[sel_metric].mean().reset_index()
        row   = means.loc[means[sel_metric].idxmax()]
        return row["Category"], row["Model"]

    best_base_cat, best_base_model = _pick_best(baseline_pt)
    best_net_cat,  best_net_model  = _pick_best(network_pt)

    print(f"\n{'=' * 80}")
    print(f"  PRESENTATION TABLES  (by {sel_metric})")
    print(f"{'=' * 80}")
    print(f"  Best baseline : {best_base_cat} / {best_base_model}")
    print(f"  Best network  : {best_net_cat}  / {best_net_model}")

    # --- Per-ticker delta ---
    base_ser = baseline_pt[
        (baseline_pt["Category"] == best_base_cat)
        & (baseline_pt["Model"] == best_base_model)
    ][["Ticker", sel_metric]].rename(columns={sel_metric: "__base"})
    net_ser = network_pt[
        (network_pt["Category"] == best_net_cat)
        & (network_pt["Model"] == best_net_model)
    ][["Ticker", sel_metric]].rename(columns={sel_metric: "__net"})
    delta_df = base_ser.merge(net_ser, on="Ticker", how="inner").copy()
    delta_df["__delta"] = delta_df["__net"] - delta_df["__base"]
    delta_df = delta_df.sort_values("__delta", ascending=False)
    top5    = delta_df.head(5)["Ticker"].tolist()
    bottom5 = delta_df.tail(5)["Ticker"].tolist()

    print(f"\n{'=' * 80}")
    print(f"  Per-Ticker Tables")
    print(f"  Baseline : {best_base_cat} / {_best_model_label(best_base_model)}")
    print(f"  Network  : {best_net_cat}  / {_best_model_label(best_net_model)}")
    print(f"{'=' * 80}")
    _present_per_ticker_table_clf(
        per_ticker, top5,
        best_base_cat, best_base_model, best_net_cat, best_net_model,
        sel_metric, "Top 5: Largest Network Improvement",
    )
    _present_per_ticker_table_clf(
        per_ticker, bottom5,
        best_base_cat, best_base_model, best_net_cat, best_net_model,
        sel_metric, "Bottom 5: Smallest Network Improvement",
    )
    if named_tickers is None:
        named_tickers = ["AAPL", "TSLA", "GOOG", "META", "MSFT", "NVDA", "NFLX", "AMZN"]
    _present_per_ticker_table_clf(
        per_ticker,
        named_tickers,
        best_base_cat, best_base_model, best_net_cat, best_net_model,
        sel_metric, "Named Tickers",
    )

    # --- Wilcoxon test ---
    print_wilcoxon_best_network_vs_baseline_clf(metrics_df)

    # --- Structural breakdown tables ---
    network_pt["_dist"]   = network_pt["Category"].map(_graph_type_label)
    network_pt["_weight"] = network_pt["Category"].map(_present_weight_scheme)
    network_pt["_struct"] = network_pt["Category"].map(_structure_cat)

    _print_structural_table_clf(
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
        sel_metric=sel_metric,
        title=f"Graph Distance Comparison  (by {sel_metric})",
    )
    _print_structural_table_clf(
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
        sel_metric=sel_metric,
        title=f"Weighting Scheme Comparison  (by {sel_metric})",
    )
    _print_structural_table_clf(
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
        sel_metric=sel_metric,
        title=f"Feature Structure Comparison  (by {sel_metric})",
    )


def print_wilcoxon_best_network_vs_baseline_clf(
    metrics_df: pd.DataFrame,
) -> None:
    """
    Compare the best network classifier against the best baseline classifier
    using paired per-ticker ROC-AUC via a Wilcoxon signed-rank test.

    Model selection is based on the highest mean per-ticker ROC-AUC.
    """
    if metrics_df.empty or "ROC_AUC" not in metrics_df.columns:
        print("\n  (Wilcoxon test skipped: ROC_AUC column not available.)")
        return

    baseline_categories = BASELINE_CATEGORIES_CLF
    per_ticker = (
        metrics_df
        .groupby(["Category", "Ticker", "Model"])
        .agg(ROC_AUC=("ROC_AUC", "mean"))
        .reset_index()
    )
    per_ticker = per_ticker.dropna(subset=["ROC_AUC"])

    baseline_df = per_ticker[per_ticker["Category"].isin(baseline_categories)].copy()
    network_df = per_ticker[~per_ticker["Category"].isin(baseline_categories)].copy()

    if baseline_df.empty or network_df.empty:
        print("\n  (Wilcoxon test skipped: need both baseline and network results.)")
        return

    # Select best models by highest mean ROC_AUC across tickers
    baseline_best = (
        baseline_df.groupby(["Category", "Model"])
        .agg(mean_ROC_AUC=("ROC_AUC", "mean"))
        .sort_values("mean_ROC_AUC", ascending=False)
        .reset_index()
        .iloc[0]
    )
    network_best = (
        network_df.groupby(["Category", "Model"])
        .agg(mean_ROC_AUC=("ROC_AUC", "mean"))
        .sort_values("mean_ROC_AUC", ascending=False)
        .reset_index()
        .iloc[0]
    )

    baseline_pair = baseline_df[
        (baseline_df["Category"] == baseline_best["Category"])
        & (baseline_df["Model"] == baseline_best["Model"])
    ][["Ticker", "ROC_AUC"]].rename(columns={"ROC_AUC": "baseline"})
    network_pair = network_df[
        (network_df["Category"] == network_best["Category"])
        & (network_df["Model"] == network_best["Model"])
    ][["Ticker", "ROC_AUC"]].rename(columns={"ROC_AUC": "network"})

    paired = baseline_pair.merge(network_pair, on="Ticker", how="inner")
    if paired.empty:
        print("\n  (Wilcoxon test skipped: no shared tickers.)")
        return

    diff = paired["network"] - paired["baseline"]
    nonzero = paired[diff != 0].copy()
    if nonzero.empty:
        print("\n  (Wilcoxon test skipped: all paired ROC_AUC differences are zero.)")
        return

    diff_nz = nonzero["network"] - nonzero["baseline"]

    test_two = wilcoxon(
        nonzero["network"].values,
        nonzero["baseline"].values,
        alternative="two-sided",
    )
    # One-sided: network > baseline (network wins if ROC_AUC is higher)
    test_greater = wilcoxon(
        nonzero["network"].values,
        nonzero["baseline"].values,
        alternative="greater",
    )

    n_net_wins = int((diff_nz > 0).sum())
    n_base_wins = int((diff_nz < 0).sum())

    print(f"\n{'=' * 80}")
    print("  Wilcoxon Signed-Rank Test: Best Network vs Best Baseline (ROC-AUC)")
    print(f"{'=' * 80}")
    print(
        f"  Best baseline : {baseline_best['Category']} / {baseline_best['Model']}  "
        f"(mean ROC_AUC = {baseline_best['mean_ROC_AUC']:.4f})"
    )
    print(
        f"  Best network  : {network_best['Category']} / {network_best['Model']}  "
        f"(mean ROC_AUC = {network_best['mean_ROC_AUC']:.4f})"
    )
    print(f"  Paired tickers (non-zero diff): {len(nonzero)}")
    print(f"  Network wins / Baseline wins: {n_net_wins} / {n_base_wins}")
    print(f"  Median(network - baseline ROC_AUC): {float(diff_nz.median()):.4f}")
    print(f"  Mean(network - baseline ROC_AUC):   {float(diff_nz.mean()):.4f}")
    print(f"  Two-sided: statistic = {float(test_two.statistic):.4f}, p = {float(test_two.pvalue):.6g}")
    print(f"  One-sided (network > baseline): p = {float(test_greater.pvalue):.6g}")


def print_presentation_report_clf(
    metrics_df_raw: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selection: str = "roc_auc",
    named_tickers: "Optional[List[str]]" = None,
) -> None:
    """Print the concise classification report used for presentation logs."""
    print_k_breakdown_clf(metrics_df_raw, selection=selection)
    print_weighting_scheme_breakdown_clf(metrics_df, selection=selection)
    print_presentation_tables_clf(
        metrics_df,
        selection=selection,
        named_tickers=named_tickers,
    )


# ---------------------------------------------------------------------------
# Load-from-disk entry point
# ---------------------------------------------------------------------------

def load_and_print_classification_results(
    results_dir: "str | Path" = None,
    sample_tickers: "Optional[List[str]]" = None,
    selection: str = "roc_auc",
    present: bool = False,
    present_only: bool = False,
) -> None:
    """
    Load saved classification JSON results and reprint all summary tables.

    Parameters
    ----------
    results_dir    : Folder containing ``classification_results.json``.
                     Defaults to ``<repo_root>/FinanceNetworks/results``.
    sample_tickers : Tickers for per-ticker tables.
                     Defaults to ``["AAPL", "TSLA", "GOOG", "META", "MSFT"]``.
    selection      : Metric used for model ranking.  One of ``'roc_auc'``
                     (default), ``'f1'``, ``'weighted_recall'``.
    present        : If True, generate condensed presentation-ready tables.
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

    sel_label = _SEL_MAP.get(selection, _SEL_MAP["roc_auc"])[0]
    print(f"Loading classification metrics from {raw_path} ...")
    print(f"Priority metric: {sel_label}")

    # Step 1: load raw fold-level metrics
    metrics_df_raw = pd.read_json(raw_path, orient="records")

    # Step 2: coalesce [k=N] categories into super-categories
    metrics_df = coalesce_categories(
        metrics_df_raw, metric_col="ROC_AUC", higher_is_better=True,
    )

    # Step 3: build summary
    summary = summarize_classification(metrics_df, selection=selection)

    if present_only:
        print_presentation_report_clf(
            metrics_df_raw,
            metrics_df,
            selection=selection,
            named_tickers=sample_tickers,
        )
    else:
        print_classification_summary(summary, title="Classification Summary (from saved results)")
        print_best_classifier(summary, selection=selection)
        print_compact_clf_leaderboard(summary, selection=selection)
        print_overall_best(summary, selection=selection)
        print_wilcoxon_best_network_vs_baseline_clf(metrics_df)
        print_per_ticker_network_vs_baseline_clf(metrics_df, sample_tickers, selection=selection)
        print_per_ticker_classification(metrics_df, sample_tickers, selection=selection)

        # Structural breakdowns (k-breakdown uses pre-coalesce data)
        print_k_breakdown_clf(metrics_df_raw, selection=selection)
        print_weighting_scheme_breakdown_clf(metrics_df, selection=selection)

        if present:
            print_presentation_tables_clf(
                metrics_df,
                selection=selection,
                named_tickers=sample_tickers,
            )

    print(f"\n  (Loaded {len(metrics_df_raw)} rows from {raw_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print classification results from saved JSON.",
    )
    parser.add_argument(
        "--selection",
        choices=["roc_auc", "f1", "weighted_recall"],
        default="roc_auc",
        help=(
            "Metric used for ranking and selecting 'best' models. "
            "'roc_auc' (default) ranks by mean ROC-AUC; "
            "'f1' ranks by mean F1; "
            "'weighted_recall' ranks by mean Weighted Recall."
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
        "--present",
        action="store_true",
        help=(
            "Generate condensed presentation-ready tables: per-ticker top/bottom-5 "
            "improvement, named-ticker table, Wilcoxon test, and structural "
            "comparison tables (distance / weighting / structure)."
        ),
    )
    parser.add_argument(
        "--present-only",
        action="store_true",
        help=(
            "Print only the presentation-oriented classification report: k breakdown, "
            "weighting-scheme breakdown, Wilcoxon, and all presentation tables."
        ),
    )
    args = parser.parse_args()
    load_and_print_classification_results(
        results_dir=args.results_dir,
        sample_tickers=args.tickers,
        selection=args.selection,
        present=args.present,
        present_only=args.present_only,
    )
