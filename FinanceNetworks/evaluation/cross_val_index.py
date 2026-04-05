"""
evaluation/cross_val_index.py
=============================
Expanding-window cross-validation for HAR / network regression models on the
Oxford-Man Realized Volatility Indices dataset.

Mirrors cross_val.py but:
  - Loads data via ``get_index_data_for_har`` (actual 5-min RV, not squared-
    return proxy).
  - Uses 21 global equity indices instead of individual US stocks.
  - Adjusts hyper-parameters for the smaller cross-section (21 tickers):
    KNN_VALUES = [1, 3, 5] (unchanged — still valid for N=21).

All evaluation, saving, and reporting functions are reused from the existing
codebase so the output format is identical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Reuse existing infrastructure ────────────────────────────────────────────
from data.preprocess_index import get_index_data_for_har, TICKERS
from data.preprocess import remove_outliers  # noqa: F401 (used by cross_val_multi internally)

from models.baselines import (
    HARLogRegressor,
    HARExtendedLogRegressor,
    ARIMALogY,
    GARCHWeeklyRV,
    DCCGARCHWeeklyRV,
    RegimeSwitchingHARLogRegressor,
)
from models.network_models import (
    NetworkHARRegressor,
    NetworkVARRegressor,
    LearnedWeightNetworkHARRegressor,
)
from models.correlation_network import (
    SquaredCorrelationNetwork,
    PartialCorrelationNetwork,
    MutualInformationNetwork,
)
from evaluation.cross_val import (
    cross_val_multi,
    save_results,
    save_prediction_store,
    select_best_on_validation,
    print_best_test_summary,
    _with_no_outlier_variants,
)
from evaluation.interpretability import (
    save_model_params,
    save_graph_snapshots,
    save_feature_snapshots,
)
from visualize.utils import coalesce_categories
from visualize.print_results import (
    summarize_benchmarks,
    print_summary,
    print_best_per_category,
    print_compact_leaderboard,
    print_per_ticker_tables,
    print_summary_excluding_outliers,
    print_wilcoxon_best_network_vs_baseline,
)


# ── Helpers (identical to cross_val.py) ──────────────────────────────────────

def _merge_pred_store(
    pred_store: dict,
    new_store: dict,
) -> None:
    """In-place merge of new_store into pred_store."""
    for t, df in new_store.items():
        new_cols = [c for c in df.columns if c != "Y_true"]
        if t in pred_store:
            pred_store[t] = pred_store[t].join(df[new_cols], how="outer")
        else:
            pred_store[t] = df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    SAMPLE_TICKERS = ["SPX2", "FTSE2", "N2252", "GDAXI2", "IXIC2"]
    RESULTS_DIR = Path(__file__).parent.parent / "results" / "index_results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    KNN_VALUES = [1, 2, 3, 4, 5]

    print("Loading and preprocessing Oxford-Man index data...")
    data_dict = get_index_data_for_har()
    tickers = list(data_dict.keys())
    print(f"  {len(tickers)} indices loaded: {tickers}")

    graph_n_jobs = max(1, min(24, (os.cpu_count() or 1) - 1))
    print(f"Using {graph_n_jobs} worker processes for graph builds.")

    # ── Offline graph builds ─────────────────────────────────────────────────
    print("\nBuilding squared-correlation networks (k=1, 3, 5)...")
    nets_sq: dict = {}
    data_dicts_net: dict = {}
    for k_val in KNN_VALUES:
        net_k = SquaredCorrelationNetwork(
            window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
            graph_type="knn", k=k_val,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_net[k_val] = net_k.fit_transform(data_dict)
        nets_sq[k_val] = net_k
        print(f"  [SqCorr] k={k_val}: {net_k.n_all_snapshots_} total, {net_k.n_snapshots_} saved.")

    GRAPHS_DIR = RESULTS_DIR / "graphs"
    FEATURES_DIR = RESULTS_DIR / "feature_snapshots"
    for k_val in KNN_VALUES:
        save_graph_snapshots(nets_sq[k_val], GRAPHS_DIR, f"sqcorr_k{k_val}")
        save_feature_snapshots(data_dicts_net[k_val], FEATURES_DIR, f"sqcorr_k{k_val}", tickers=SAMPLE_TICKERS)

    print("\nBuilding partial-correlation networks (k=1, 3, 5)...")
    nets_pcorr: dict = {}
    data_dicts_pcorr: dict = {}
    for k_val in KNN_VALUES:
        net_pk = PartialCorrelationNetwork(
            window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
            graph_type="knn", k=k_val, shrinkage=0.1,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_pcorr[k_val] = net_pk.fit_transform(data_dict)
        nets_pcorr[k_val] = net_pk
        print(f"  [PCorr]  k={k_val}: {net_pk.n_all_snapshots_} total, {net_pk.n_snapshots_} saved.")

    for k_val in KNN_VALUES:
        save_graph_snapshots(nets_pcorr[k_val], GRAPHS_DIR, f"pcorr_k{k_val}")
        save_feature_snapshots(data_dicts_pcorr[k_val], FEATURES_DIR, f"pcorr_k{k_val}", tickers=SAMPLE_TICKERS)

    print("\nBuilding exp-kernel networks (k=1, 3, 5)...")
    data_dicts_exp: dict = {}
    for k_val in KNN_VALUES:
        net_exp = SquaredCorrelationNetwork(
            window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
            graph_type="knn", k=k_val,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            idw_kernel="exp", exp_lambda=5.0,
        )
        data_dicts_exp[k_val] = net_exp.fit_transform(data_dict)
        print(f"  [ExpKernel] k={k_val}: {net_exp.n_all_snapshots_} total, {net_exp.n_snapshots_} saved.")
        save_feature_snapshots(data_dicts_exp[k_val], FEATURES_DIR, f"expkernel_k{k_val}", tickers=SAMPLE_TICKERS)

    print("\nBuilding mutual-information networks (k=1, 3, 5)...")
    nets_mi: dict = {}
    data_dicts_mi: dict = {}
    for k_val in KNN_VALUES:
        net_mi = MutualInformationNetwork(
            window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
            graph_type="knn", k=k_val, n_bins=10,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_mi[k_val] = net_mi.fit_transform(data_dict)
        nets_mi[k_val] = net_mi
        print(f"  [MI]     k={k_val}: {net_mi.n_all_snapshots_} total, {net_mi.n_snapshots_} saved.")

    for k_val in KNN_VALUES:
        save_graph_snapshots(nets_mi[k_val], GRAPHS_DIR, f"mi_k{k_val}")
        save_feature_snapshots(data_dicts_mi[k_val], FEATURES_DIR, f"mi_k{k_val}", tickers=SAMPLE_TICKERS)

    # ── Model catalogues ─────────────────────────────────────────────────────
    baseline_catalogue: Dict[str, Dict[str, Any]] = {
        "HAR": {
            "HAR":                        (HARLogRegressor(use_market=False),                               False),
            "HAR (no outliers)":          (HARLogRegressor(use_market=False),                               True),
            "HAR-Extended":               (HARExtendedLogRegressor(use_market=False),                       False),
            "HAR-Extended (no outliers)": (HARExtendedLogRegressor(use_market=False),                       True),
            "HAR-Lasso (a=0.01)":         (HARLogRegressor(lasso_alpha=0.01, use_market=False),             False),
            "HAR-Lasso (a=0.1)":          (HARLogRegressor(lasso_alpha=0.1, use_market=False),              False),
            "HAR-Ext-Lasso (a=0.01)":     (HARExtendedLogRegressor(lasso_alpha=0.01, use_market=False),     False),
            "HAR-Ext-Lasso (a=0.1)":      (HARExtendedLogRegressor(lasso_alpha=0.1, use_market=False),      False),
        },
        "ARIMA": {
            "ARIMA(1,0,1)":               (ARIMALogY(order=(1, 0, 1)),                      False),
            "ARIMA(2,0,1)":               (ARIMALogY(order=(2, 0, 1)),                      False),
            "ARIMA(1,0,2)":               (ARIMALogY(order=(1, 0, 2)),                      False),
            "ARIMA(2,0,2)":               (ARIMALogY(order=(2, 0, 2)),                      False),
            "ARIMA(1,1,0)":               (ARIMALogY(order=(1, 1, 0)),                      False),
            "ARIMA(0,1,1)":               (ARIMALogY(order=(0, 1, 1)),                      False),
            "ARIMA(1,1,1)":               (ARIMALogY(order=(1, 1, 1)),                      False),
            "ARIMA(2,1,1)":               (ARIMALogY(order=(2, 1, 1)),                      False),
            "ARIMA(1,1,2)":               (ARIMALogY(order=(1, 1, 2)),                      False),
        },
        "GARCH": {
            "GARCH(1,1)":                 (GARCHWeeklyRV(p=1, q=1, horizon=5, returns_multiplier=100.0),              False),
            "GARCH(2,1)":                 (GARCHWeeklyRV(p=2, q=1, horizon=5, returns_multiplier=100.0),              False),
            "GARCH(1,2)":                 (GARCHWeeklyRV(p=1, q=2, horizon=5, returns_multiplier=100.0),              False),
            "GARCH(2,2)":                 (GARCHWeeklyRV(p=2, q=2, horizon=5, returns_multiplier=100.0),              False),
            "GARCH(3,2)":                 (GARCHWeeklyRV(p=3, q=2, horizon=5, returns_multiplier=100.0),              False),
            "GARCH(2,3)":                 (GARCHWeeklyRV(p=2, q=3, horizon=5, returns_multiplier=100.0),              False),
            "GARCH(3,3)":                 (GARCHWeeklyRV(p=3, q=3, horizon=5, returns_multiplier=100.0),              False),
            "DCC-GARCH(1,1)":             (DCCGARCHWeeklyRV(p=1, q=1, horizon=5, aux_returns_col=None, returns_multiplier=100.0),  False),
        },
        "RegimeSwitching": {
            "RegHAR (p50)": (RegimeSwitchingHARLogRegressor(regime_percentile=0.50, use_market=False), False),
            "RegHAR (p75)": (RegimeSwitchingHARLogRegressor(regime_percentile=0.75, use_market=False), False),
            "RegHAR (p90)": (RegimeSwitchingHARLogRegressor(regime_percentile=0.90, use_market=False), False),
            "RegHAR-Lasso (p50)": (RegimeSwitchingHARLogRegressor(lasso_alpha=0.01, regime_percentile=0.50, use_market=False), False),
            "RegHAR-Lasso (p75)": (RegimeSwitchingHARLogRegressor(lasso_alpha=0.01, regime_percentile=0.75, use_market=False), False),
        },
    }

    def _network_models() -> Dict[str, Any]:
        return _with_no_outlier_variants({
            "NetHAR (Lasso a=0.20)":          (NetworkHARRegressor(lasso_alpha=0.20, use_market=False),                                          False),
            "NetHAR (Lasso a=0.10)":          (NetworkHARRegressor(lasso_alpha=0.10, use_market=False),                                          False),
            "NetHAR (Lasso a=0.05)":          (NetworkHARRegressor(lasso_alpha=0.05, use_market=False),                                          False),
            "NetHAR (Lasso a=0.01)":          (NetworkHARRegressor(lasso_alpha=0.01, use_market=False),                                          False),
            "NetHAR (OLS)":                   (NetworkHARRegressor(lasso_alpha=0.0, use_market=False),                                           False),
            "NetHAR (Ridge a=0.01)":          (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=0.01, use_market=False),                         False),
            "NetHAR (Ridge a=0.10)":          (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=0.10, use_market=False),                         False),
            "NetHAR (Ridge a=1.0)":           (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=1.0, use_market=False),                          False),
            "NetworkVAR (a=0.0, b=0.5)":      (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=0.5, use_market=False),                   False),
            "NetworkVAR (a=0.0, b=1.0)":      (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=1.0, use_market=False),                   False),
            "NetworkVAR (a=0.0, b=None)":     (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=None, use_market=False),                  False),
            "NetworkVAR (a=0.1, b=0.5)":      (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=0.5, use_market=False),                   False),
            "NetworkVAR (a=0.5, b=0.5)":      (NetworkVARRegressor(stage2_alpha=0.5,  correction_bound=0.5, use_market=False),                   False),
            "NetworkVAR (a=0.1, b=1.0)":      (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=1.0, use_market=False),                   False),
            "NetworkVAR (a=0.1, b=None)":     (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=None, use_market=False),                  False),
        })

    def _network_models_clustering() -> Dict[str, Any]:
        return _with_no_outlier_variants({
            "NetHAR+C (Lasso a=0.20)":        (NetworkHARRegressor(lasso_alpha=0.20, use_clustering=True, use_market=False),                    False),
            "NetHAR+C (Lasso a=0.10)":        (NetworkHARRegressor(lasso_alpha=0.10, use_clustering=True, use_market=False),                    False),
            "NetHAR+C (Lasso a=0.05)":        (NetworkHARRegressor(lasso_alpha=0.05, use_clustering=True, use_market=False),                    False),
            "NetHAR+C (Lasso a=0.01)":        (NetworkHARRegressor(lasso_alpha=0.01, use_clustering=True, use_market=False),                    False),
            "NetHAR+C (OLS)":                 (NetworkHARRegressor(lasso_alpha=0.0,  use_clustering=True, use_market=False),                    False),
            "NetHAR+C (Ridge a=0.01)":        (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.01, use_clustering=True, use_market=False),  False),
            "NetHAR+C (Ridge a=0.10)":        (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.10, use_clustering=True, use_market=False),  False),
            "NetHAR+C (Ridge a=1.0)":         (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=1.0, use_clustering=True, use_market=False),   False),
            "NetworkVAR+C (a=0.0, b=0.5)":    (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=0.5, use_clustering=True, use_market=False),  False),
            "NetworkVAR+C (a=0.0, b=1.0)":    (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=1.0, use_clustering=True, use_market=False),  False),
            "NetworkVAR+C (a=0.0, b=None)":   (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=None, use_clustering=True, use_market=False), False),
            "NetworkVAR+C (a=0.1, b=0.5)":    (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=0.5, use_clustering=True, use_market=False),  False),
            "NetworkVAR+C (a=0.5, b=0.5)":    (NetworkVARRegressor(stage2_alpha=0.5,  correction_bound=0.5, use_clustering=True, use_market=False),  False),
            "NetworkVAR+C (a=0.1, b=1.0)":    (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=1.0, use_clustering=True, use_market=False),  False),
            "NetworkVAR+C (a=0.1, b=None)":   (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=None, use_clustering=True, use_market=False), False),
        })

    def _network_models_sign_split() -> Dict[str, Any]:
        return _with_no_outlier_variants({
            "NetHAR-Split (Lasso a=0.20)":     (NetworkHARRegressor(lasso_alpha=0.20, use_sign_split=True, use_market=False),                  False),
            "NetHAR-Split (Lasso a=0.10)":     (NetworkHARRegressor(lasso_alpha=0.10, use_sign_split=True, use_market=False),                  False),
            "NetHAR-Split (Lasso a=0.05)":     (NetworkHARRegressor(lasso_alpha=0.05, use_sign_split=True, use_market=False),                  False),
            "NetHAR-Split (Lasso a=0.01)":     (NetworkHARRegressor(lasso_alpha=0.01, use_sign_split=True, use_market=False),                  False),
            "NetHAR-Split (OLS)":              (NetworkHARRegressor(lasso_alpha=0.0,  use_sign_split=True, use_market=False),                  False),
            "NetHAR-Split (Ridge a=0.01)":     (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.01, use_sign_split=True, use_market=False),False),
            "NetHAR-Split (Ridge a=0.10)":     (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.10, use_sign_split=True, use_market=False),False),
            "NetHAR-Split (Ridge a=1.0)":      (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=1.0, use_sign_split=True, use_market=False), False),
            "NetworkVAR-Split (a=0.0, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=0.5, use_sign_split=True, use_market=False),  False),
            "NetworkVAR-Split (a=0.0, b=None)":(NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=None, use_sign_split=True, use_market=False), False),
            "NetworkVAR-Split (a=0.1, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=0.5, use_sign_split=True, use_market=False),  False),
            "NetworkVAR-Split (a=0.1, b=1.0)": (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=1.0, use_sign_split=True, use_market=False),  False),
            "NetworkVAR-Split (a=0.1, b=None)":(NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=None, use_sign_split=True, use_market=False), False),
        })

    def _network_models_sign_split_clustering() -> Dict[str, Any]:
        return _with_no_outlier_variants({
            "NetHAR+CSplit (Lasso a=0.20)":    (NetworkHARRegressor(lasso_alpha=0.20, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit (Lasso a=0.01)":    (NetworkHARRegressor(lasso_alpha=0.01, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit (Lasso a=0.05)":    (NetworkHARRegressor(lasso_alpha=0.05, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit (OLS)":             (NetworkHARRegressor(lasso_alpha=0.0,  use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit (Lasso a=0.1)":     (NetworkHARRegressor(lasso_alpha=0.1,  use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit (Ridge a=0.01)":    (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.01, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit (Ridge a=0.10)":    (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.10, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit (Ridge a=1.0)":     (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=1.0, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetworkVAR+CSplit (a=0.0,b=0.5)": (NetworkVARRegressor(stage2_alpha=0.0, correction_bound=0.5, use_clustering=True, use_sign_split=True, use_market=False),  False),
            "NetworkVAR+CSplit (a=0.0,b=None)":(NetworkVARRegressor(stage2_alpha=0.0, correction_bound=None, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetworkVAR+CSplit (a=0.1,b=0.5)": (NetworkVARRegressor(stage2_alpha=0.1, correction_bound=0.5, use_clustering=True, use_sign_split=True, use_market=False),  False),
            "NetworkVAR+CSplit (a=0.1,b=1.0)": (NetworkVARRegressor(stage2_alpha=0.1, correction_bound=1.0, use_clustering=True, use_sign_split=True, use_market=False),  False),
            "NetworkVAR+CSplit (a=0.1,b=None)":(NetworkVARRegressor(stage2_alpha=0.1, correction_bound=None, use_clustering=True, use_sign_split=True, use_market=False), False),
        })

    def _learned_weight_models(k_val: int) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        for m_val in range(1, k_val):
            models[f"LearnedW (m={m_val}, Ridge a=0.01)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=0.01, use_market=False), False)
            models[f"LearnedW (m={m_val}, Ridge a=1.0)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=1.0, use_market=False), False)
            models[f"LearnedW (m={m_val}, Ridge a=0.1)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=0.1, use_market=False), False)
            models[f"LearnedW (m={m_val}, Lasso a=0.01)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, lasso_alpha=0.01, use_market=False), False)
            models[f"LearnedW (m={m_val}, Lasso a=0.05)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, lasso_alpha=0.05, use_market=False), False)
        return _with_no_outlier_variants(models)

    def _learned_weight_clustering_models(k_val: int) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        for m_val in range(1, k_val):
            models[f"LearnedW+C (m={m_val}, Ridge a=0.01)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=0.01, use_clustering=True, use_market=False), False)
            models[f"LearnedW+C (m={m_val}, Ridge a=1.0)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=1.0, use_clustering=True, use_market=False), False)
            models[f"LearnedW+C (m={m_val}, Ridge a=0.1)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=0.1, use_clustering=True, use_market=False), False)
            models[f"LearnedW+C (m={m_val}, Lasso a=0.01)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, lasso_alpha=0.01, use_clustering=True, use_market=False), False)
            models[f"LearnedW+C (m={m_val}, Lasso a=0.05)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, lasso_alpha=0.05, use_clustering=True, use_market=False), False)
        return _with_no_outlier_variants(models)

    # ── Run baselines ────────────────────────────────────────────────────────
    print(f"\nRunning baseline models on {len(tickers)} indices...")
    metrics_df, pred_store, all_params = cross_val_multi(
        data_dict, baseline_catalogue, tickers,
        n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
    )
    all_net_metrics: list = []

    # ── Squared-correlation network models ───────────────────────────────────
    print("\nRunning squared-correlation network models (k=1..5)...")
    for k_val in KNN_VALUES:
        net_catalogue: Dict[str, Dict[str, Any]] = {
            f"Network [k={k_val}]": _network_models()
        }
        dd_net = data_dicts_net[k_val]
        metrics_k, pred_store_k, params_k = cross_val_multi(
            dd_net, net_catalogue, list(dd_net.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_k)
        all_params.extend(params_k)
        _merge_pred_store(pred_store, pred_store_k)

    # ── Partial-correlation network models ───────────────────────────────────
    print("\nRunning partial-correlation network models (k=1..5)...")
    for k_val in KNN_VALUES:
        pcorr_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr Network [k={k_val}]": _network_models()
        }
        dd_pc = data_dicts_pcorr[k_val]
        metrics_pk, pred_store_pk, params_pk = cross_val_multi(
            dd_pc, pcorr_catalogue, list(dd_pc.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_pk)
        all_params.extend(params_pk)
        _merge_pred_store(pred_store, pred_store_pk)

    # ── Exp-kernel network models ────────────────────────────────────────────
    print("\nRunning exp-kernel network models (k=1..5)...")
    for k_val in KNN_VALUES:
        exp_catalogue: Dict[str, Dict[str, Any]] = {
            f"ExpKernel [k={k_val}]": _network_models()
        }
        dd_exp = data_dicts_exp[k_val]
        metrics_ek, pred_store_ek, params_ek = cross_val_multi(
            dd_exp, exp_catalogue, list(dd_exp.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_ek)
        all_params.extend(params_ek)
        _merge_pred_store(pred_store, pred_store_ek)

    # ── Clustering-feature network models ────────────────────────────────────
    print("\nRunning clustering-feature network models (k=1..5)...")
    for k_val in KNN_VALUES:
        clust_catalogue: Dict[str, Dict[str, Any]] = {
            f"Clustering [k={k_val}]": _network_models_clustering()
        }
        dd_sq = data_dicts_net[k_val]
        metrics_cl, pred_store_cl, params_cl = cross_val_multi(
            dd_sq, clust_catalogue, list(dd_sq.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_cl)
        all_params.extend(params_cl)
        _merge_pred_store(pred_store, pred_store_cl)

    # ── Mutual-information network models ────────────────────────────────────
    print("\nRunning mutual-information network models (k=1..5)...")
    for k_val in KNN_VALUES:
        mi_catalogue: Dict[str, Dict[str, Any]] = {
            f"MI Network [k={k_val}]": _network_models()
        }
        dd_mi = data_dicts_mi[k_val]
        metrics_mi, pred_store_mi, params_mi = cross_val_multi(
            dd_mi, mi_catalogue, list(dd_mi.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_mi)
        all_params.extend(params_mi)
        _merge_pred_store(pred_store, pred_store_mi)

    # ── Exp + clustering ─────────────────────────────────────────────────────
    print("\nRunning exp-kernel + clustering network models (k=1..5)...")
    for k_val in KNN_VALUES:
        expc_catalogue: Dict[str, Dict[str, Any]] = {
            f"Exp+Clustering [k={k_val}]": _network_models_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        metrics_ec, pred_store_ec, params_ec = cross_val_multi(
            dd_exp, expc_catalogue, list(dd_exp.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_ec)
        all_params.extend(params_ec)
        _merge_pred_store(pred_store, pred_store_ec)

    # ── Sign-split feature benchmarks ────────────────────────────────────────
    print("\nRunning sign-split feature benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        split_catalogue: Dict[str, Dict[str, Any]] = {
            f"SplitFeatures [k={k_val}]": _network_models_sign_split()
        }
        dd_sq = data_dicts_net[k_val]
        metrics_sp, pred_store_sp, params_sp = cross_val_multi(
            dd_sq, split_catalogue, list(dd_sq.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_sp)
        all_params.extend(params_sp)
        _merge_pred_store(pred_store, pred_store_sp)

    # ── Sign-split + clustering ──────────────────────────────────────────────
    print("\nRunning sign-split + clustering benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        splitc_catalogue: Dict[str, Dict[str, Any]] = {
            f"Split+Clustering [k={k_val}]": _network_models_sign_split_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        metrics_sc, pred_store_sc, params_sc = cross_val_multi(
            dd_exp, splitc_catalogue, list(dd_exp.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_sc)
        all_params.extend(params_sc)
        _merge_pred_store(pred_store, pred_store_sc)

    # ── PCorr sign-split ─────────────────────────────────────────────────────
    print("\nRunning PCorr sign-split benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        pcorr_split_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr Split [k={k_val}]": _network_models_sign_split()
        }
        dd_pc = data_dicts_pcorr[k_val]
        metrics_ps, pred_store_ps, params_ps = cross_val_multi(
            dd_pc, pcorr_split_catalogue, list(dd_pc.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_ps)
        all_params.extend(params_ps)
        _merge_pred_store(pred_store, pred_store_ps)

    # ── ExpKernel sign-split ─────────────────────────────────────────────────
    print("\nRunning exp-kernel sign-split benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        exp_split_catalogue: Dict[str, Dict[str, Any]] = {
            f"ExpKernel Split [k={k_val}]": _network_models_sign_split()
        }
        dd_exp = data_dicts_exp[k_val]
        metrics_es, pred_store_es, params_es = cross_val_multi(
            dd_exp, exp_split_catalogue, list(dd_exp.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_es)
        all_params.extend(params_es)
        _merge_pred_store(pred_store, pred_store_es)

    # ── MI sign-split ────────────────────────────────────────────────────────
    print("\nRunning MI sign-split benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        mi_split_catalogue: Dict[str, Dict[str, Any]] = {
            f"MI Split [k={k_val}]": _network_models_sign_split()
        }
        dd_mi = data_dicts_mi[k_val]
        metrics_ms, pred_store_ms, params_ms = cross_val_multi(
            dd_mi, mi_split_catalogue, list(dd_mi.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_ms)
        all_params.extend(params_ms)
        _merge_pred_store(pred_store, pred_store_ms)

    # ── Learned-weight models (SqCorr) ───────────────────────────────────────
    print("\nRunning learned-weight models (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lw_catalogue: Dict[str, Dict[str, Any]] = {
            f"LearnedWeight [k={k_val}]": _learned_weight_models(k_val)
        }
        dd_sq = data_dicts_net[k_val]
        metrics_lw, pred_store_lw, params_lw = cross_val_multi(
            dd_sq, lw_catalogue, list(dd_sq.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lw)
        all_params.extend(params_lw)
        _merge_pred_store(pred_store, pred_store_lw)

    # ── Learned-weight models (PCorr) ────────────────────────────────────────
    print("\nRunning learned-weight models (PCorr, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lw_pc_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr LearnedWeight [k={k_val}]": _learned_weight_models(k_val)
        }
        dd_pc = data_dicts_pcorr[k_val]
        metrics_lwp, pred_store_lwp, params_lwp = cross_val_multi(
            dd_pc, lw_pc_catalogue, list(dd_pc.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lwp)
        all_params.extend(params_lwp)
        _merge_pred_store(pred_store, pred_store_lwp)

    # ── Learned-weight models (MI) ───────────────────────────────────────────
    print("\nRunning learned-weight models (MI, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lw_mi_catalogue: Dict[str, Dict[str, Any]] = {
            f"MI LearnedWeight [k={k_val}]": _learned_weight_models(k_val)
        }
        dd_mi = data_dicts_mi[k_val]
        metrics_lwm, pred_store_lwm, params_lwm = cross_val_multi(
            dd_mi, lw_mi_catalogue, list(dd_mi.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lwm)
        all_params.extend(params_lwm)
        _merge_pred_store(pred_store, pred_store_lwm)

    # ── Learned-weight + clustering (SqCorr) ─────────────────────────────────
    print("\nRunning learned-weight + clustering models (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lwc_catalogue: Dict[str, Dict[str, Any]] = {
            f"LW+Clustering [k={k_val}]": _learned_weight_clustering_models(k_val)
        }
        dd_sq = data_dicts_net[k_val]
        metrics_lwc, pred_store_lwc, params_lwc = cross_val_multi(
            dd_sq, lwc_catalogue, list(dd_sq.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lwc)
        all_params.extend(params_lwc)
        _merge_pred_store(pred_store, pred_store_lwc)

    # ── Learned-weight + clustering (PCorr) ──────────────────────────────────
    print("\nRunning learned-weight + clustering (PCorr, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lwc_pc_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr LW+Clustering [k={k_val}]": _learned_weight_clustering_models(k_val)
        }
        dd_pc = data_dicts_pcorr[k_val]
        metrics_lwcp, pred_store_lwcp, params_lwcp = cross_val_multi(
            dd_pc, lwc_pc_catalogue, list(dd_pc.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lwcp)
        all_params.extend(params_lwcp)
        _merge_pred_store(pred_store, pred_store_lwcp)

    # ── Learned-weight + clustering (MI) ─────────────────────────────────────
    print("\nRunning learned-weight + clustering (MI, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lwc_mi_catalogue: Dict[str, Dict[str, Any]] = {
            f"MI LW+Clustering [k={k_val}]": _learned_weight_clustering_models(k_val)
        }
        dd_mi = data_dicts_mi[k_val]
        metrics_lwcm, pred_store_lwcm, params_lwcm = cross_val_multi(
            dd_mi, lwc_mi_catalogue, list(dd_mi.keys()),
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lwcm)
        all_params.extend(params_lwcm)
        _merge_pred_store(pred_store, pred_store_lwcm)

    # ── Aggregate and save ───────────────────────────────────────────────────
    metrics_df = pd.concat([metrics_df] + all_net_metrics, ignore_index=True)

    save_results(metrics_df, summarize_benchmarks(metrics_df), RESULTS_DIR)
    save_model_params(all_params, RESULTS_DIR, "regression_model_params.json")
    save_prediction_store(pred_store, RESULTS_DIR)

    # Coalesce k-variants for printing
    metrics_coalesced = coalesce_categories(metrics_df)

    # ── Validation / Test reporting ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VALIDATION-FOLD SUMMARY  (Fold 1 — used for model selection)")
    print("=" * 80)
    val_metrics = metrics_coalesced[metrics_coalesced["Fold"] == 1]
    val_summary = summarize_benchmarks(val_metrics)
    print_summary(val_summary, title="Validation Summary (all models)")
    print_best_per_category(val_summary)

    print("\n" + "=" * 80)
    print("TEST-FOLD SUMMARY  (Fold 0 — held-out final evaluation)")
    print("=" * 80)
    test_metrics = metrics_coalesced[metrics_coalesced["Fold"] == 0]
    test_summary = summarize_benchmarks(test_metrics)
    print_summary(test_summary, title="Test Summary (all models)")
    print_best_per_category(test_summary)

    best_models = select_best_on_validation(metrics_coalesced, val_fold=1)
    print("\n" + "=" * 80)
    print("BEST MODEL PER CATEGORY  (chosen on validation fold)")
    print("=" * 80)
    print(best_models.to_string(index=False))

    print_best_test_summary(metrics_coalesced, best_models, test_fold=0)

    summary = summarize_benchmarks(metrics_coalesced)
    print_summary(summary, title="Full Summary — both folds (all indices)")
    print_compact_leaderboard(summary)
    print_summary_excluding_outliers(metrics_coalesced, r2_threshold=-1.0)
    print_wilcoxon_best_network_vs_baseline(metrics_coalesced)
    print_per_ticker_tables(metrics_coalesced, SAMPLE_TICKERS)


if __name__ == "__main__":
    main()
