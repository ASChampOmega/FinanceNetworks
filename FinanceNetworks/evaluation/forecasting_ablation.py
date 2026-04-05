"""
evaluation/forecasting_ablation.py
===================================
Systematic ablation study for regression (forecasting) results.

Reads the saved fold-level results JSON and computes the best model per
ablation dimension.  The four ablation axes are:

1. **k value** — for each distance metric, what k ∈ {1,2,3,4,5} is best?
2. **Feature structure** — for each distance metric, which feature combination
   (Plain / Clustering / Split / Split+Clustering) is best?
3. **Distance metric** — for each feature structure, which distance metric
   (Squared Correlation / Partial Correlation / Mutual Information) is best?
4. **Weighting scheme** — IDW vs Exp-Kernel vs Learned Weights.

Run
---
    python -m evaluation.forecasting_ablation [--results-dir results]
    python -m evaluation.forecasting_ablation --results-dir results/index_results
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Category → ablation-dimension decoders
# ---------------------------------------------------------------------------

_K_RE = re.compile(r"\[k=(\d+)\]")

BASELINE_CATEGORIES = frozenset({"HAR", "ARIMA", "GARCH", "RegimeSwitching"})


def _extract_k(cat: str) -> int | None:
    m = _K_RE.search(cat)
    return int(m.group(1)) if m else None


def _extract_distance(cat: str) -> str | None:
    """Distance metric used for the graph."""
    if cat in BASELINE_CATEGORIES:
        return None
    c = re.sub(r"\s*\[k=\d+\]$", "", cat)
    if c.startswith("PCorr"):
        return "Partial Correlation"
    if c.startswith("MI ") or c == "MI Network" or c.startswith("MI Split"):
        return "Mutual Information"
    if c.startswith("ExpKernel"):
        return "Squared Correlation"
    # Everything else uses SqCorr data
    return "Squared Correlation"


def _extract_weighting(cat: str) -> str | None:
    if cat in BASELINE_CATEGORIES:
        return None
    c = re.sub(r"\s*\[k=\d+\]$", "", cat)
    if "LearnedWeight" in c or "LW+" in c:
        return "Learned"
    if "ExpKernel" in c or "Exp+" in c:
        return "Exp"
    return "IDW"


def _extract_structure(cat: str) -> str | None:
    if cat in BASELINE_CATEGORIES:
        return None
    c = re.sub(r"\s*\[k=\d+\]$", "", cat)
    if "Split+Clustering" in c or "+CSplit" in c or "CSplit" in c:
        return "Split+Clustering"
    if "Split" in c:
        return "Split"
    if "Clustering" in c or "+C" in c:
        return "Clustering"
    return "Plain"


def decode_category(cat: str) -> dict:
    return {
        "k": _extract_k(cat),
        "distance": _extract_distance(cat),
        "weighting": _extract_weighting(cat),
        "structure": _extract_structure(cat),
    }


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def load_and_decode(results_path: Path) -> pd.DataFrame:
    """Load fold-level results and add ablation columns."""
    df = pd.read_json(results_path)
    decoded = df["Category"].apply(decode_category).apply(pd.Series)
    df = pd.concat([df, decoded], axis=1)
    return df


def _best_model_in_group(
    group_df: pd.DataFrame,
    metric: str = "RMSE_log",
    lower_is_better: bool = True,
) -> pd.Series:
    """Return the best (Category, Model) row within a group."""
    agg = (
        group_df.groupby(["Category", "Model"])[metric]
        .mean()
        .reset_index()
    )
    if lower_is_better:
        best_idx = agg[metric].idxmin()
    else:
        best_idx = agg[metric].idxmax()
    return agg.loc[best_idx]


def ablation_table(
    df: pd.DataFrame,
    group_col: str,
    metric: str = "RMSE_log",
    lower_is_better: bool = True,
    filter_col: str | None = None,
    filter_val: str | None = None,
) -> pd.DataFrame:
    """
    Build an ablation table: for each unique value of *group_col*, find the
    best (Category, Model) by *metric* averaged over (Ticker, Fold).

    Optionally filter to rows where *filter_col* == *filter_val* first.
    """
    sub = df[df[group_col].notna()].copy()
    if filter_col and filter_val is not None:
        sub = sub[sub[filter_col] == filter_val]
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for val, g in sub.groupby(group_col):
        best = _best_model_in_group(g, metric, lower_is_better)
        rows.append({
            group_col: val,
            "Best Category": best["Category"],
            "Best Model": best["Model"],
            f"mean_{metric}": best[metric],
        })
    out = pd.DataFrame(rows).sort_values(f"mean_{metric}",
                                          ascending=lower_is_better)
    return out.reset_index(drop=True)


def full_ablation(
    df: pd.DataFrame,
    metric: str = "RMSE_log",
    lower_is_better: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run all four ablation axes and return a dict of DataFrames.

    Returns
    -------
    {
        "k_by_distance": DataFrame,
        "structure_by_distance": DataFrame,
        "distance_by_structure": DataFrame,
        "weighting": DataFrame,
    }
    """
    net_df = df[df["k"].notna()].copy()  # exclude baselines

    results: dict[str, pd.DataFrame] = {}

    # 1. Best k for each distance metric
    rows_k = []
    for dist, g_dist in net_df.groupby("distance"):
        for k_val, g_k in g_dist.groupby("k"):
            best = _best_model_in_group(g_k, metric, lower_is_better)
            rows_k.append({
                "distance": dist,
                "k": int(k_val),
                "Best Category": best["Category"],
                "Best Model": best["Model"],
                f"mean_{metric}": best[metric],
            })
    results["k_by_distance"] = pd.DataFrame(rows_k)

    # 2. Best feature structure for each distance metric
    rows_s = []
    for dist, g_dist in net_df.groupby("distance"):
        for struct, g_s in g_dist.groupby("structure"):
            best = _best_model_in_group(g_s, metric, lower_is_better)
            rows_s.append({
                "distance": dist,
                "structure": struct,
                "Best Category": best["Category"],
                "Best Model": best["Model"],
                f"mean_{metric}": best[metric],
            })
    results["structure_by_distance"] = pd.DataFrame(rows_s)

    # 3. Best distance metric for each feature structure
    rows_d = []
    for struct, g_s in net_df.groupby("structure"):
        for dist, g_d in g_s.groupby("distance"):
            best = _best_model_in_group(g_d, metric, lower_is_better)
            rows_d.append({
                "structure": struct,
                "distance": dist,
                "Best Category": best["Category"],
                "Best Model": best["Model"],
                f"mean_{metric}": best[metric],
            })
    results["distance_by_structure"] = pd.DataFrame(rows_d)

    # 4. Weighting scheme comparison
    rows_w = []
    for w, g_w in net_df.groupby("weighting"):
        best = _best_model_in_group(g_w, metric, lower_is_better)
        rows_w.append({
            "weighting": w,
            "Best Category": best["Category"],
            "Best Model": best["Model"],
            f"mean_{metric}": best[metric],
        })
    results["weighting"] = pd.DataFrame(rows_w)

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_ablation(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("ablation_regression_*.csv"):
        stale.unlink()
    for name, tbl in tables.items():
        path = out_dir / f"ablation_regression_{name}.csv"
        tbl.to_csv(path, index=False)
        print(f"  → {path}")


def _default_results_dirs() -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    return [repo_root / "results", repo_root / "results" / "index_results"]


def _label_for_results_dir(results_dir: Path) -> str:
    return "index" if results_dir.name == "index_results" else "stock"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Forecasting ablation study")
    parser.add_argument("--results-dir", action="append", default=None,
                        help="Directory containing results_bench.json. Repeat to process multiple folders. Defaults to stock + index.")
    parser.add_argument("--metric", type=str, default="RMSE_log",
                        choices=["RMSE_log", "MAE_log", "R2_log", "RMSE", "MAE", "R2"],
                        help="Metric for ranking models (default: RMSE_log)")
    args = parser.parse_args()

    metric = args.metric
    lower_is_better = metric not in ("R2", "R2_log")

    results_dirs = [Path(p) for p in args.results_dir] if args.results_dir else _default_results_dirs()
    for results_dir in results_dirs:
        results_path = results_dir / "results_bench.json"
        if not results_path.exists():
            print(f"Skipping {results_dir} — missing {results_path.name}")
            continue

        dataset_label = _label_for_results_dir(results_dir)
        print(f"Loading {results_path} ...")
        df = load_and_decode(results_path)
        print(f"  [{dataset_label}] {len(df)} rows, {df['Category'].nunique()} categories, "
              f"{df['Ticker'].nunique()} tickers")

        tables = full_ablation(df, metric=metric, lower_is_better=lower_is_better)

        print(f"\n{'='*80}")
        print(f"  FORECASTING ABLATION STUDY [{dataset_label.upper()}]  (metric: {metric})")
        print(f"{'='*80}")

        print(f"\n── Best k per distance metric ──")
        print(tables["k_by_distance"].to_string(index=False))

        print(f"\n── Best feature structure per distance metric ──")
        print(tables["structure_by_distance"].to_string(index=False))

        print(f"\n── Best distance metric per feature structure ──")
        print(tables["distance_by_structure"].to_string(index=False))

        print(f"\n── Weighting scheme comparison ──")
        print(tables["weighting"].to_string(index=False))

        baselines = df[df["k"].isna()].copy()
        if not baselines.empty:
            best_bl = _best_model_in_group(baselines, metric, lower_is_better)
            net_overall = _best_model_in_group(
                df[df["k"].notna()], metric, lower_is_better)
            print(f"\n── Baseline vs Network ──")
            print(f"  Best baseline : {best_bl['Category']} / {best_bl['Model']}  "
                  f"({metric} = {best_bl[metric]:.6f})")
            print(f"  Best network  : {net_overall['Category']} / {net_overall['Model']}  "
                  f"({metric} = {net_overall[metric]:.6f})")

        out_dir = results_dir / "ablation"
        save_ablation(tables, out_dir)
        print(f"\nCSV tables saved to {out_dir}")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
