"""
evaluation/classification_ablation.py
======================================
Systematic ablation study for classification results.

Same four ablation axes as the forecasting variant, but uses classification
metrics (ROC_AUC, F1, Precision, Recall).

Run
---
    python -m evaluation.classification_ablation [--results-dir results]
    python -m evaluation.classification_ablation --results-dir results/index_results
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

CLF_BASELINE_CATEGORIES = frozenset({
    "HAR-Logit", "HAR-Ext-Logit", "RegimeSwitching-Logit", "DCC-GARCH-Logit",
})


def _extract_k(cat: str) -> int | None:
    m = _K_RE.search(cat)
    return int(m.group(1)) if m else None


def _extract_distance(cat: str) -> str | None:
    if cat in CLF_BASELINE_CATEGORIES:
        return None
    c = re.sub(r"\s*\[k=\d+\]$", "", cat)
    if c.startswith("PCorr"):
        return "Partial Correlation"
    if c.startswith("MI ") or c == "MI Network" or c.startswith("MI Split"):
        return "Mutual Information"
    if c.startswith("ExpKernel") or c.startswith("Exp+"):
        return "Squared Correlation"
    return "Squared Correlation"


def _extract_weighting(cat: str) -> str | None:
    if cat in CLF_BASELINE_CATEGORIES:
        return None
    c = re.sub(r"\s*\[k=\d+\]$", "", cat)
    if "LearnedWeight" in c or "LW+" in c:
        return "Learned"
    if "ExpKernel" in c or "Exp+" in c:
        return "Exp"
    return "IDW"


def _extract_structure(cat: str) -> str | None:
    if cat in CLF_BASELINE_CATEGORIES:
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
# Aggregation (same logic as forecasting_ablation but default metric differs)
# ---------------------------------------------------------------------------

def load_and_decode(results_path: Path) -> pd.DataFrame:
    df = pd.read_json(results_path)
    decoded = df["Category"].apply(decode_category).apply(pd.Series)
    df = pd.concat([df, decoded], axis=1)
    return df


def _best_model_in_group(
    group_df: pd.DataFrame,
    metric: str = "ROC_AUC",
    lower_is_better: bool = False,
) -> pd.Series:
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


def full_ablation(
    df: pd.DataFrame,
    metric: str = "ROC_AUC",
    lower_is_better: bool = False,
) -> dict[str, pd.DataFrame]:
    net_df = df[df["k"].notna()].copy()
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


def save_ablation(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("ablation_classification_*.csv"):
        stale.unlink()
    for name, tbl in tables.items():
        path = out_dir / f"ablation_classification_{name}.csv"
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
    parser = argparse.ArgumentParser(description="Classification ablation study")
    parser.add_argument("--results-dir", action="append", default=None,
                        help="Directory containing classification_results.json. Repeat to process multiple folders. Defaults to stock + index.")
    parser.add_argument("--metric", type=str, default="ROC_AUC",
                        choices=["ROC_AUC", "F1", "Precision", "Recall",
                                 "Weighted_Recall", "Accuracy"],
                        help="Metric for ranking models (default: ROC_AUC)")
    args = parser.parse_args()

    metric = args.metric
    lower_is_better = False  # all classification metrics are higher=better

    results_dirs = [Path(p) for p in args.results_dir] if args.results_dir else _default_results_dirs()
    for results_dir in results_dirs:
        results_path = results_dir / "classification_results.json"
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
        print(f"  CLASSIFICATION ABLATION STUDY [{dataset_label.upper()}]  (metric: {metric})")
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
