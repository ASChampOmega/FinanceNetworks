"""
evaluation/sanity_runner.py
===========================
Lightweight smoke-test runner for all 4 experiment pipelines.

Runs a minimal model catalogue on a small subset of tickers to verify the
full pipeline works end-to-end without spending hours on the real run.

Usage (from FinanceNetworks/):
    python -m evaluation.sanity_runner --task stock-regression   --results-dir /tmp/sanity/stock_reg
    python -m evaluation.sanity_runner --task stock-classification --results-dir /tmp/sanity/stock_clf
    python -m evaluation.sanity_runner --task index-regression   --results-dir /tmp/sanity/index_reg
    python -m evaluation.sanity_runner --task index-classification --results-dir /tmp/sanity/index_clf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in headless environments
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.cross_val import (
    cross_val_multi,
    save_results,
    save_prediction_store,
    expanding_folds,
)
from evaluation.classification import (
    classification_cv_multi,
    save_classification_results,
    save_classification_prediction_store,
)
from visualize.print_results import summarize_benchmarks, print_summary
from visualize.print_results_classification import (
    summarize_classification,
    print_classification_summary,
)
from visualize.plot_model_results import (
    plot_ticker_predictions,
    plot_summary_metrics,
)

# ── Shared constants ─────────────────────────────────────────────────────────
STOCK_SAMPLE = ["AAPL", "MSFT"]
INDEX_SAMPLE = ["SPX2", "FTSE2"]
K_VAL = 3
N_STOCK_TICKERS = 35  # load a small universe so graphs are cheaper


def _graph_n_jobs() -> int:
    return max(1, min(4, (os.cpu_count() or 1) - 1))

def _save_clf_summary_plot(summary_df: "pd.DataFrame", save_path: "Path") -> None:
    """Save a simple ROC-AUC bar chart for a classification summary."""
    roc_col = "mean_ROC_AUC" if "mean_ROC_AUC" in summary_df.columns else None
    if roc_col is None:
        # Try to find any column containing ROC
        roc_cols = [c for c in summary_df.columns if "ROC" in c.upper()]
        roc_col = roc_cols[0] if roc_cols else None
    if roc_col is None:
        # No ROC metric — save a placeholder so the file still exists
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No ROC_AUC metric available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Classification Summary")
        fig.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return
    data = summary_df[roc_col].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.4)))
    ax.barh([str(i) for i in data.index], data.values, color="steelblue")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, label="Baseline (0.5)")
    ax.set_xlabel(roc_col)
    ax.set_title("Classification Summary — Mean ROC-AUC", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Stock regression ─────────────────────────────────────────────────────────

def run_stock_regression(results_dir: Path) -> None:
    from data import get_data_for_har
    from models.baselines import (
        HARExtendedLogRegressor,
        ARIMALogY,
        GARCHWeeklyRV,
    )
    from models.network_models import NetworkHARRegressor
    from models.correlation_network import SquaredCorrelationNetwork

    results_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = _graph_n_jobs()

    print("Loading data ...")
    data_dict = get_data_for_har(N_STOCK_TICKERS)
    tickers = [t for t in data_dict if t in STOCK_SAMPLE] or list(data_dict)[:2]
    print(f"  Tickers: {tickers}")

    # One graph build
    print(f"Building graph (k={K_VAL}) ...")
    net = SquaredCorrelationNetwork(
        window=60, step=1, save_step=5, n_jobs=n_jobs,
        graph_type="knn", k=K_VAL,
        feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
    )
    dd_net = net.fit_transform(data_dict)

    # Minimal catalogue
    catalogue: Dict[str, Dict[str, Any]] = {
        "HAR": {"HAR-Extended": (HARExtendedLogRegressor(), False)},
        "ARIMA": {"ARIMA(1,1,1)": (ARIMALogY(order=(1, 1, 1)), False)},
        "GARCH": {"GARCH(1,1)": (GARCHWeeklyRV(p=1, q=1, horizon=5), False)},
    }
    print("Running baseline models ...")
    m_base, ps = cross_val_multi(
        data_dict, catalogue, tickers,
        n_splits=1, sample_tickers=STOCK_SAMPLE,
    )

    net_catalogue: Dict[str, Dict[str, Any]] = {
        f"Network [k={K_VAL}]": {
            "NetHAR (Lasso a=0.05)": (NetworkHARRegressor(lasso_alpha=0.05), False),
        },
    }
    print("Running network model ...")
    m_net, ps_net = cross_val_multi(
        dd_net, net_catalogue, list(dd_net)[:len(tickers)],
        n_splits=1, sample_tickers=STOCK_SAMPLE,
    )
    for t, df in ps_net.items():
        cols = [c for c in df.columns if c != "Y_true"]
        if t in ps:
            ps[t] = ps[t].join(df[cols], how="outer")

    metrics_df = pd.concat([m_base, m_net], ignore_index=True)
    summary = summarize_benchmarks(metrics_df)
    save_results(metrics_df, summary, results_dir)
    save_prediction_store(ps, results_dir)
    print_summary(summary, title="Sanity — Stock Regression")

    print("Generating plots ...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_ticker_predictions(ps, metrics_df, STOCK_SAMPLE, save_dir=str(plots_dir))
    plot_summary_metrics(summary, save_path=str(plots_dir / "summary_metrics.png"))
    print("Stock regression sanity PASSED ✓")


# ── Stock classification ─────────────────────────────────────────────────────

def run_stock_classification(results_dir: Path) -> None:
    from data import get_data_for_har
    from models.baselines_classification import (
        HARLogitClassifier,
        HARExtendedLogitClassifier,
    )
    from models.network_models_classification import NetworkHARClassifier
    from models.correlation_network import SquaredCorrelationNetwork

    results_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = _graph_n_jobs()

    print("Loading data ...")
    data_dict = get_data_for_har(N_STOCK_TICKERS)
    tickers = [t for t in data_dict if t in STOCK_SAMPLE] or list(data_dict)[:2]

    print(f"Building graph (k={K_VAL}) ...")
    net = SquaredCorrelationNetwork(
        window=60, step=1, save_step=5, n_jobs=n_jobs,
        graph_type="knn", k=K_VAL,
        feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
    )
    dd_net = net.fit_transform(data_dict)

    catalogue: Dict[str, Dict[str, Any]] = {
        "HAR": {
            "Logit-HAR": (HARLogitClassifier(), False),
            "Logit-HAR-Ext": (HARExtendedLogitClassifier(), False),
        },
    }
    print("Running baseline classifiers ...")
    m_base, ps = classification_cv_multi(
        data_dict, catalogue, tickers,
        n_splits=1, sample_tickers=STOCK_SAMPLE,
    )

    net_catalogue: Dict[str, Dict[str, Any]] = {
        f"Network [k={K_VAL}]": {
            "NetHAR-Clf (C=1.0)": (NetworkHARClassifier(C=1.0), False),
        },
    }
    print("Running network classifier ...")
    m_net, ps_net = classification_cv_multi(
        dd_net, net_catalogue, list(dd_net)[:len(tickers)],
        n_splits=1, sample_tickers=STOCK_SAMPLE,
    )
    for t, df in ps_net.items():
        cols = [c for c in df.columns if c != "Y_true_spike"]
        if t in ps:
            ps[t] = ps[t].join(df[cols], how="outer")

    metrics_df = pd.concat([m_base, m_net], ignore_index=True)
    summary = summarize_classification(metrics_df)
    save_classification_results(metrics_df, summary, results_dir)
    save_classification_prediction_store(ps, results_dir)
    print_classification_summary(summary, title="Sanity — Stock Classification")

    print("Generating plots ...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _save_clf_summary_plot(summary, plots_dir / "summary_roc_auc.png")
    print("Stock classification sanity PASSED ✓")


# ── Index regression ─────────────────────────────────────────────────────────

def run_index_regression(results_dir: Path) -> None:
    from data.preprocess_index import get_index_data_for_har
    from models.baselines import (
        HARExtendedLogRegressor,
        ARIMALogY,
        GARCHWeeklyRV,
    )
    from models.network_models import NetworkHARRegressor
    from models.correlation_network import SquaredCorrelationNetwork

    results_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = _graph_n_jobs()

    print("Loading index data ...")
    data_dict = get_index_data_for_har()
    tickers = [t for t in data_dict if t in INDEX_SAMPLE] or list(data_dict)[:2]
    print(f"  Tickers: {tickers}")

    print(f"Building graph (k={K_VAL}) ...")
    net = SquaredCorrelationNetwork(
        window=60, step=1, save_step=5, n_jobs=n_jobs,
        graph_type="knn", k=K_VAL,
        feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
    )
    dd_net = net.fit_transform(data_dict)

    catalogue: Dict[str, Dict[str, Any]] = {
        "HAR": {"HAR-Extended": (HARExtendedLogRegressor(use_market=False), False)},
        "ARIMA": {"ARIMA(1,1,1)": (ARIMALogY(order=(1, 1, 1)), False)},
        "GARCH": {"GARCH(1,1)": (GARCHWeeklyRV(p=1, q=1, horizon=5), False)},
    }
    print("Running baseline models ...")
    m_base, ps = cross_val_multi(
        data_dict, catalogue, tickers,
        n_splits=1, sample_tickers=INDEX_SAMPLE,
    )

    net_catalogue: Dict[str, Dict[str, Any]] = {
        f"Network [k={K_VAL}]": {
            "NetHAR (Lasso a=0.05)": (NetworkHARRegressor(lasso_alpha=0.05, use_market=False), False),
        },
    }
    print("Running network model ...")
    m_net, ps_net = cross_val_multi(
        dd_net, net_catalogue, list(dd_net)[:len(tickers)],
        n_splits=1, sample_tickers=INDEX_SAMPLE,
    )
    for t, df in ps_net.items():
        cols = [c for c in df.columns if c != "Y_true"]
        if t in ps:
            ps[t] = ps[t].join(df[cols], how="outer")

    metrics_df = pd.concat([m_base, m_net], ignore_index=True)
    summary = summarize_benchmarks(metrics_df)
    save_results(metrics_df, summary, results_dir)
    save_prediction_store(ps, results_dir)
    print_summary(summary, title="Sanity — Index Regression")

    print("Generating plots ...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_ticker_predictions(ps, metrics_df, INDEX_SAMPLE, save_dir=str(plots_dir))
    plot_summary_metrics(summary, save_path=str(plots_dir / "summary_metrics.png"))
    print("Index regression sanity PASSED ✓")


# ── Index classification ─────────────────────────────────────────────────────

def run_index_classification(results_dir: Path) -> None:
    from data.preprocess_index import get_index_data_for_har
    from models.baselines_classification import (
        HARLogitClassifier,
        HARExtendedLogitClassifier,
    )
    from models.network_models_classification import NetworkHARClassifier
    from models.correlation_network import SquaredCorrelationNetwork

    results_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = _graph_n_jobs()

    print("Loading index data ...")
    data_dict = get_index_data_for_har()
    # Use all index tickers — the index universe is small (21) and using just
    # 2 tickers leads to degenerate folds with a single class at q=0.75.
    tickers = list(data_dict.keys())
    print(f"  Tickers ({len(tickers)}): {tickers}")

    print(f"Building graph (k={K_VAL}) ...")
    net = SquaredCorrelationNetwork(
        window=60, step=1, save_step=5, n_jobs=n_jobs,
        graph_type="knn", k=K_VAL,
        feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
    )
    dd_net = net.fit_transform(data_dict)

    catalogue: Dict[str, Dict[str, Any]] = {
        "HAR": {
            "Logit-HAR": (HARLogitClassifier(use_market=False), False),
            "Logit-HAR-Ext": (HARExtendedLogitClassifier(use_market=False), False),
        },
    }
    print("Running baseline classifiers ...")
    m_base, ps = classification_cv_multi(
        data_dict, catalogue, tickers,
        n_splits=1, spike_quantile=0.75,
        sample_tickers=INDEX_SAMPLE,
    )

    net_catalogue: Dict[str, Dict[str, Any]] = {
        f"Network [k={K_VAL}]": {
            "NetHAR-Clf (C=1.0)": (NetworkHARClassifier(C=1.0, use_market=False), False),
        },
    }
    print("Running network classifier ...")
    m_net, ps_net = classification_cv_multi(
        dd_net, net_catalogue, list(dd_net.keys()),
        n_splits=1, spike_quantile=0.75,
        sample_tickers=INDEX_SAMPLE,
    )
    for t, df in ps_net.items():
        cols = [c for c in df.columns if c != "Y_true_spike"]
        if t in ps:
            ps[t] = ps[t].join(df[cols], how="outer")

    metrics_df = pd.concat([m_base, m_net], ignore_index=True)
    summary = summarize_classification(metrics_df)
    save_classification_results(metrics_df, summary, results_dir)
    save_classification_prediction_store(ps, results_dir)
    print_classification_summary(summary, title="Sanity — Index Classification")

    print("Generating plots ...")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _save_clf_summary_plot(summary, plots_dir / "summary_roc_auc.png")
    print("Index classification sanity PASSED ✓")


# ── CLI ──────────────────────────────────────────────────────────────────────

TASKS = {
    "stock-regression":      run_stock_regression,
    "stock-classification":  run_stock_classification,
    "index-regression":      run_index_regression,
    "index-classification":  run_index_classification,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check runner for all experiment pipelines.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASKS),
        help="Which experiment to smoke-test.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory to write throwaway results into.",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    TASKS[args.task](results_dir)


if __name__ == "__main__":
    main()
