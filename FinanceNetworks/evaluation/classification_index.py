"""
evaluation/classification_index.py
===================================
Expanding-window cross-validation for volatility-spike classifiers on the
Oxford-Man Realized Volatility Indices dataset.

Mirrors classification.py but:
  - Loads data via ``get_index_data_for_har`` (actual 5-min RV).
  - Uses 21 global equity indices instead of individual US stocks.
  - All evaluation / saving / reporting functions are reused from the
    existing codebase.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Reuse existing infrastructure ────────────────────────────────────────────
from data.preprocess_index import get_index_data_for_har

from models.baselines_classification import (
    HARLogitClassifier,
    HARExtendedLogitClassifier,
    RegimeSwitchingHARLogitClassifier,
    DCCGARCHSpikeClassifier,
)
from models.network_models_classification import (
    NetworkHARClassifier,
    NetworkVARClassifier,
    LearnedWeightNetworkHARClassifier,
)
from models.correlation_network import (
    SquaredCorrelationNetwork,
    PartialCorrelationNetwork,
    MutualInformationNetwork,
)
from evaluation.cross_val import expanding_folds, _with_no_outlier_variants, select_best_on_validation
from evaluation.classification import (
    classification_cv_multi,
    save_classification_results,
    save_classification_prediction_store,
    eval_classification,
    print_best_clf_test_summary,
)
from evaluation.interpretability import (
    save_model_params,
    save_graph_snapshots,
    save_feature_snapshots,
)
from visualize.utils import coalesce_categories
from visualize.print_results_classification import (
    summarize_classification,
    print_classification_summary,
    print_best_classifier,
    print_compact_clf_leaderboard,
    print_overall_best,
    print_per_ticker_classification,
    print_wilcoxon_best_network_vs_baseline_clf,
    print_per_ticker_network_vs_baseline_clf,
    print_k_breakdown_clf,
    print_weighting_scheme_breakdown_clf,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _merge_clf_pred_store(
    pred_store: dict,
    new_store: dict,
) -> None:
    """In-place merge of new_store into pred_store."""
    for t, df in new_store.items():
        new_cols = [c for c in df.columns if c != "Y_true_spike"]
        if t in pred_store:
            pred_store[t] = pred_store[t].join(df[new_cols], how="outer")
        else:
            pred_store[t] = df


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Index classification evaluation")
    parser.add_argument(
        "--use-validation-split",
        action="store_true",
        help="Run two expanding folds (validation + test). Default is a single held-out test fold.",
    )
    args = parser.parse_args()
    n_splits = 2 if args.use_validation_split else 1

    SAMPLE_TICKERS = ["SPX2", "FTSE2", "N2252", "GDAXI2", "IXIC2"]
    RESULTS_DIR = Path(__file__).parent.parent / "results" / "index_results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    KNN_VALUES = [1, 2, 3, 4, 5]
    INDEX_SPIKE_QUANTILE = 0.75
    INDEX_SPIKE_LOOKBACK = 252 * 3

    print("Loading and preprocessing Oxford-Man index data...")
    data_dict = get_index_data_for_har()
    tickers = list(data_dict.keys())
    print(f"  {len(tickers)} indices loaded: {tickers}")
    print(
        "Index spike labelling uses the last "
        f"{INDEX_SPIKE_LOOKBACK} training days at the "
        f"{INDEX_SPIKE_QUANTILE:.0%} quantile."
    )

    graph_n_jobs = max(1, min(8, (os.cpu_count() or 1) - 1))
    cv_n_jobs = max(1, min(24, (os.cpu_count() or 1) - 1))
    print(f"Using {graph_n_jobs} graph workers, {cv_n_jobs} CV workers.")
    print(f"Using {n_splits} expanding-window fold(s) for evaluation.")

    # ── Offline graph builds — reuse cache from cross_val_index if available ─
    from data.graph_cache import load_graph_data, graph_cache_exists
    CACHE_DIR = RESULTS_DIR / "graph_cache"
    _tags = ["sqcorr", "pcorr", "exp", "mi"]
    _all_cached = all(graph_cache_exists(CACHE_DIR, t, KNN_VALUES) for t in _tags)
    cache_loaded = False

    if _all_cached:
        print("\nLoading cached graph features from index regression run...")
        data_dicts_net, nets_sq = load_graph_data(CACHE_DIR, "sqcorr", KNN_VALUES)
        data_dicts_pcorr, nets_pc = load_graph_data(CACHE_DIR, "pcorr", KNN_VALUES)
        data_dicts_exp, nets_exp = load_graph_data(CACHE_DIR, "exp", KNN_VALUES)
        data_dicts_mi, nets_mi = load_graph_data(CACHE_DIR, "mi", KNN_VALUES)
        cache_loaded = True
        print("  Graph cache loaded — skipping expensive fit_transform.")
    else:
        print("\nGraph cache not found — building from scratch...")
        print("  (Run cross_val_index.py first to populate the cache.)")
        data_dicts_net: dict = {}
        nets_sq: dict = {}
        for k_val in KNN_VALUES:
            net_k = SquaredCorrelationNetwork(
                window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
                graph_type="knn", k=k_val,
                feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            )
            data_dicts_net[k_val] = net_k.fit_transform(data_dict)
            nets_sq[k_val] = net_k
            print(f"  [SqCorr] k={k_val}: {net_k.n_all_snapshots_} total, {net_k.n_snapshots_} saved.")

        data_dicts_pcorr: dict = {}
        nets_pc: dict = {}
        for k_val in KNN_VALUES:
            net_pk = PartialCorrelationNetwork(
                window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
                graph_type="knn", k=k_val, shrinkage=0.1,
                feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            )
            data_dicts_pcorr[k_val] = net_pk.fit_transform(data_dict)
            nets_pc[k_val] = net_pk
            print(f"  [PCorr]  k={k_val}: {net_pk.n_all_snapshots_} total, {net_pk.n_snapshots_} saved.")

        data_dicts_exp: dict = {}
        nets_exp: dict = {}
        for k_val in KNN_VALUES:
            net_exp = SquaredCorrelationNetwork(
                window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
                graph_type="knn", k=k_val,
                feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
                idw_kernel="exp", exp_lambda=5.0,
            )
            data_dicts_exp[k_val] = net_exp.fit_transform(data_dict)
            nets_exp[k_val] = net_exp
            print(f"  [ExpKernel] k={k_val}: {net_exp.n_all_snapshots_} total, {net_exp.n_snapshots_} saved.")

        data_dicts_mi: dict = {}
        nets_mi: dict = {}
        for k_val in KNN_VALUES:
            net_mi = MutualInformationNetwork(
                window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
                graph_type="knn", k=k_val, n_bins=10,
                feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            )
            data_dicts_mi[k_val] = net_mi.fit_transform(data_dict)
            nets_mi[k_val] = net_mi
            print(f"  [MI]     k={k_val}: {net_mi.n_all_snapshots_} total, {net_mi.n_snapshots_} saved.")

        from data.graph_cache import save_graph_data

        print("\nSaving graph cache for future index classification runs...")
        save_graph_data(data_dicts_net, nets_sq, CACHE_DIR, "sqcorr")
        save_graph_data(data_dicts_pcorr, nets_pc, CACHE_DIR, "pcorr")
        save_graph_data(data_dicts_exp, nets_exp, CACHE_DIR, "exp")
        save_graph_data(data_dicts_mi, nets_mi, CACHE_DIR, "mi")
        print("  Graph cache saved to", CACHE_DIR)

    if cache_loaded:
        print("\nSkipping classification graph snapshot export; cached regression artifacts already exist.")
    else:
        GRAPHS_DIR = RESULTS_DIR / "graphs"
        FEATURES_DIR = RESULTS_DIR / "feature_snapshots"
        for k_val in KNN_VALUES:
            save_graph_snapshots(nets_sq[k_val], GRAPHS_DIR, f"clf_sqcorr_k{k_val}")
            save_graph_snapshots(nets_pc[k_val], GRAPHS_DIR, f"clf_pcorr_k{k_val}")
            save_graph_snapshots(nets_exp[k_val], GRAPHS_DIR, f"clf_expkernel_k{k_val}")
            save_graph_snapshots(nets_mi[k_val], GRAPHS_DIR, f"clf_mi_k{k_val}")
            save_feature_snapshots(data_dicts_net[k_val], FEATURES_DIR, f"clf_sqcorr_k{k_val}", tickers=SAMPLE_TICKERS)
            save_feature_snapshots(data_dicts_pcorr[k_val], FEATURES_DIR, f"clf_pcorr_k{k_val}", tickers=SAMPLE_TICKERS)
            save_feature_snapshots(data_dicts_exp[k_val], FEATURES_DIR, f"clf_expkernel_k{k_val}", tickers=SAMPLE_TICKERS)
            save_feature_snapshots(data_dicts_mi[k_val], FEATURES_DIR, f"clf_mi_k{k_val}", tickers=SAMPLE_TICKERS)

    # ── Model catalogues ─────────────────────────────────────────────────────
    baseline_catalogue: Dict[str, Dict[str, Any]] = {
        "HAR-Logit": {
            "HAR-Logit (C=0.01)":        (HARLogitClassifier(C=0.01, use_market=False),             False),
            "HAR-Logit (C=0.1)":         (HARLogitClassifier(C=0.1, use_market=False),              False),
            "HAR-Logit (C=1.0)":         (HARLogitClassifier(C=1.0, use_market=False),              False),
            "HAR-Logit (C=10.0)":        (HARLogitClassifier(C=10.0, use_market=False),             False),
            "HAR-Logit (C=1.0, no-out)": (HARLogitClassifier(C=1.0, use_market=False),              True),
        },
        "HAR-Ext-Logit": {
            "HAR-Ext-Logit (C=0.01)":        (HARExtendedLogitClassifier(C=0.01, use_market=False),  False),
            "HAR-Ext-Logit (C=0.1)":         (HARExtendedLogitClassifier(C=0.1, use_market=False),   False),
            "HAR-Ext-Logit (C=1.0)":         (HARExtendedLogitClassifier(C=1.0, use_market=False),   False),
            "HAR-Ext-Logit (C=10.0)":        (HARExtendedLogitClassifier(C=10.0, use_market=False),  False),
            "HAR-Ext-Logit (C=1.0, no-out)": (HARExtendedLogitClassifier(C=1.0, use_market=False),   True),
        },
        "RegimeSwitching-Logit": {
            "RegHAR-Logit (p50)":            (RegimeSwitchingHARLogitClassifier(regime_percentile=0.5, use_market=False),  False),
            "RegHAR-Logit (p75)":            (RegimeSwitchingHARLogitClassifier(regime_percentile=0.75, use_market=False), False),
            "RegHAR-Logit (p50, C=0.1)":     (RegimeSwitchingHARLogitClassifier(C=0.1, regime_percentile=0.5, use_market=False), False),
        },
        "DCC-GARCH-Logit": {
            "DCC-GARCH-Logit (C=0.1)":       (DCCGARCHSpikeClassifier(C=0.1, aux_returns_col=None, returns_multiplier=100.0), False),
            "DCC-GARCH-Logit (C=1.0)":       (DCCGARCHSpikeClassifier(C=1.0, aux_returns_col=None, returns_multiplier=100.0), False),
        },
    }

    def _network_clf_models() -> Dict[str, Any]:
        return _with_no_outlier_variants({
            "NetHAR-Logit (C=0.1)":           (NetworkHARClassifier(C=0.1, use_market=False),             False),
            "NetHAR-Logit (C=1.0)":           (NetworkHARClassifier(C=1.0, use_market=False),             False),
            "NetHAR-Logit (C=10.0)":          (NetworkHARClassifier(C=10.0, use_market=False),            False),
            "NetHAR-Logit (C=100.0)":         (NetworkHARClassifier(C=100.0, use_market=False),           False),
            "NetVAR-Logit (C=1,a=0.1,b=1)":   (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.1,  correction_bound=1.0, use_market=False), False),
            "NetVAR-Logit (C=1,a=0.5,b=2)":   (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.5,  correction_bound=2.0, use_market=False), False),
            "NetVAR-Logit (C=1,a=1.0,b=2)":   (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=1.0,  correction_bound=2.0, use_market=False), False),
            "NetVAR-Logit (C=1,a=0.01,b=1)":  (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.01, correction_bound=1.0, use_market=False), False),
            "NetVAR-Logit (C=10,a=0.1,b=1)":  (NetworkVARClassifier(C_stage1=10.0, stage2_alpha=0.1,  correction_bound=1.0, use_market=False), False),
            "NetVAR-Logit (C=10,a=0.5,b=2)":  (NetworkVARClassifier(C_stage1=10.0, stage2_alpha=0.5,  correction_bound=2.0, use_market=False), False),
            "NetVAR-Logit (C=10,a=1.0,b=2)":  (NetworkVARClassifier(C_stage1=10.0, stage2_alpha=1.0,  correction_bound=2.0, use_market=False), False),
        })

    def _network_clf_models_clustering() -> Dict[str, Any]:
        return _with_no_outlier_variants({
            "NetHAR+C-Logit (C=0.1)":          (NetworkHARClassifier(C=0.1,   use_clustering=True, use_market=False), False),
            "NetHAR+C-Logit (C=1.0)":          (NetworkHARClassifier(C=1.0,   use_clustering=True, use_market=False), False),
            "NetHAR+C-Logit (C=10.0)":         (NetworkHARClassifier(C=10.0,  use_clustering=True, use_market=False), False),
            "NetHAR+C-Logit (C=100.0)":        (NetworkHARClassifier(C=100.0, use_clustering=True, use_market=False), False),
            "NetVAR+C-Logit (C=1,a=0.1,b=1)":  (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.1,  correction_bound=1.0, use_clustering=True, use_market=False), False),
            "NetVAR+C-Logit (C=1,a=0.5,b=2)":  (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.5,  correction_bound=2.0, use_clustering=True, use_market=False), False),
            "NetVAR+C-Logit (C=10,a=0.1,b=1)": (NetworkVARClassifier(C_stage1=10.0, stage2_alpha=0.1,  correction_bound=1.0, use_clustering=True, use_market=False), False),
            "NetVAR+C-Logit (C=10,a=0.5,b=2)": (NetworkVARClassifier(C_stage1=10.0, stage2_alpha=0.5,  correction_bound=2.0, use_clustering=True, use_market=False), False),
        })

    def _network_clf_models_sign_split() -> Dict[str, Any]:
        return _with_no_outlier_variants({
            "NetHAR-Split-Logit (C=0.1)":          (NetworkHARClassifier(C=0.1,   use_sign_split=True, use_market=False), False),
            "NetHAR-Split-Logit (C=1.0)":          (NetworkHARClassifier(C=1.0,   use_sign_split=True, use_market=False), False),
            "NetHAR-Split-Logit (C=10.0)":         (NetworkHARClassifier(C=10.0,  use_sign_split=True, use_market=False), False),
            "NetHAR-Split-Logit (C=100.0)":        (NetworkHARClassifier(C=100.0, use_sign_split=True, use_market=False), False),
            "NetVAR-Split-Logit (C=1,a=0.1,b=1)":  (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.1,  correction_bound=1.0, use_sign_split=True, use_market=False), False),
            "NetVAR-Split-Logit (C=1,a=0.5,b=2)":  (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.5,  correction_bound=2.0, use_sign_split=True, use_market=False), False),
            "NetVAR-Split-Logit (C=10,a=0.1,b=1)": (NetworkVARClassifier(C_stage1=10.0, stage2_alpha=0.1,  correction_bound=1.0, use_sign_split=True, use_market=False), False),
        })

    def _network_clf_models_sign_split_clustering() -> Dict[str, Any]:
        return _with_no_outlier_variants({
            "NetHAR+CSplit-Logit (C=0.1)":          (NetworkHARClassifier(C=0.1,   use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit-Logit (C=1.0)":          (NetworkHARClassifier(C=1.0,   use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetHAR+CSplit-Logit (C=10.0)":         (NetworkHARClassifier(C=10.0,  use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetVAR+CSplit-Logit (C=1,a=0.1,b=1)":  (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.1,  correction_bound=1.0, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetVAR+CSplit-Logit (C=1,a=0.5,b=2)":  (NetworkVARClassifier(C_stage1=1.0,  stage2_alpha=0.5,  correction_bound=2.0, use_clustering=True, use_sign_split=True, use_market=False), False),
            "NetVAR+CSplit-Logit (C=10,a=0.1,b=1)": (NetworkVARClassifier(C_stage1=10.0, stage2_alpha=0.1,  correction_bound=1.0, use_clustering=True, use_sign_split=True, use_market=False), False),
        })

    def _learned_weight_clf_models(k_val: int) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        for m_val in range(1, k_val):
            models[f"LearnedW-Logit (m={m_val}, C=0.1)"]  = (LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=0.1, use_market=False),   False)
            models[f"LearnedW-Logit (m={m_val}, C=1.0)"]  = (LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=1.0, use_market=False),   False)
            models[f"LearnedW-Logit (m={m_val}, C=10.0)"] = (LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=10.0, use_market=False),  False)
            models[f"LearnedW-Logit (m={m_val}, C=100.0)"]= (LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=100.0, use_market=False), False)
        return _with_no_outlier_variants(models)

    def _learned_weight_clf_clustering_models(k_val: int) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        for m_val in range(1, k_val):
            models[f"LearnedW+C-Logit (m={m_val}, C=0.1)"]  = (LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=0.1,  use_clustering=True, use_market=False), False)
            models[f"LearnedW+C-Logit (m={m_val}, C=1.0)"]  = (LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=1.0,  use_clustering=True, use_market=False), False)
            models[f"LearnedW+C-Logit (m={m_val}, C=10.0)"] = (LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=10.0, use_clustering=True, use_market=False), False)
            models[f"LearnedW+C-Logit (m={m_val}, C=100.0)"]= (LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=100.0, use_clustering=True, use_market=False), False)
        return _with_no_outlier_variants(models)

    # ── Run baselines ────────────────────────────────────────────────────────
    print(f"\nRunning baseline classifiers on {len(tickers)} indices...")
    metrics_df, pred_store, all_params = classification_cv_multi(
        data_dict, baseline_catalogue, tickers,
        n_splits=n_splits,
        sample_tickers=SAMPLE_TICKERS,
        spike_quantile=INDEX_SPIKE_QUANTILE,
        spike_lookback=INDEX_SPIKE_LOOKBACK,
        save_params=True, num_workers=cv_n_jobs,
    )
    all_extra_metrics: list = []

    # ── Squared-correlation network classifiers ──────────────────────────────
    print("\nRunning network classifiers (SqCorr, k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        net_cat: Dict[str, Dict[str, Any]] = {
            f"Network [k={k_val}]": _network_clf_models()
        }
        dd_net = data_dicts_net[k_val]
        m_k, ps_k, params_k = classification_cv_multi(
            dd_net, net_cat, list(dd_net.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_k)
        all_params.extend(params_k)
        _merge_clf_pred_store(pred_store, ps_k)

    # ── Partial-correlation network classifiers ──────────────────────────────
    print("\nRunning network classifiers (PCorr, k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        pc_cat: Dict[str, Dict[str, Any]] = {
            f"PCorr Network [k={k_val}]": _network_clf_models()
        }
        dd_pc = data_dicts_pcorr[k_val]
        m_pk, ps_pk, params_pk = classification_cv_multi(
            dd_pc, pc_cat, list(dd_pc.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_pk)
        all_params.extend(params_pk)
        _merge_clf_pred_store(pred_store, ps_pk)

    # ── Exp-kernel network classifiers ───────────────────────────────────────
    print("\nRunning network classifiers (ExpKernel, k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        exp_cat: Dict[str, Dict[str, Any]] = {
            f"ExpKernel [k={k_val}]": _network_clf_models()
        }
        dd_exp = data_dicts_exp[k_val]
        m_ek, ps_ek, params_ek = classification_cv_multi(
            dd_exp, exp_cat, list(dd_exp.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_ek)
        all_params.extend(params_ek)
        _merge_clf_pred_store(pred_store, ps_ek)

    # ── SqCorr + clustering classifiers ──────────────────────────────────────
    print("\nRunning clustering-feature classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        cl_cat: Dict[str, Dict[str, Any]] = {
            f"Clustering [k={k_val}]": _network_clf_models_clustering()
        }
        dd_sq = data_dicts_net[k_val]
        m_cl, ps_cl, params_cl = classification_cv_multi(
            dd_sq, cl_cat, list(dd_sq.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_cl)
        all_params.extend(params_cl)
        _merge_clf_pred_store(pred_store, ps_cl)

    # ── Exp + clustering classifiers ─────────────────────────────────────────
    print("\nRunning exp-kernel + clustering classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        ec_cat: Dict[str, Dict[str, Any]] = {
            f"Exp+Clustering [k={k_val}]": _network_clf_models_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        m_ec, ps_ec, params_ec = classification_cv_multi(
            dd_exp, ec_cat, list(dd_exp.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_ec)
        all_params.extend(params_ec)
        _merge_clf_pred_store(pred_store, ps_ec)

    # ── Mutual-information network classifiers ───────────────────────────────
    print("\nRunning MI network classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        mi_cat: Dict[str, Dict[str, Any]] = {
            f"MI Network [k={k_val}]": _network_clf_models()
        }
        dd_mi = data_dicts_mi[k_val]
        m_mi, ps_mi, params_mi = classification_cv_multi(
            dd_mi, mi_cat, list(dd_mi.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_mi)
        all_params.extend(params_mi)
        _merge_clf_pred_store(pred_store, ps_mi)

    # ── Sign-split classifiers ───────────────────────────────────────────────
    print("\nRunning sign-split feature classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        split_cat: Dict[str, Dict[str, Any]] = {
            f"SplitFeatures [k={k_val}]": _network_clf_models_sign_split()
        }
        dd_sq = data_dicts_net[k_val]
        m_sp, ps_sp, params_sp = classification_cv_multi(
            dd_sq, split_cat, list(dd_sq.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_sp)
        all_params.extend(params_sp)
        _merge_clf_pred_store(pred_store, ps_sp)

    # ── Sign-split + clustering ──────────────────────────────────────────────
    print("\nRunning sign-split + clustering classifiers (k=1..5)...")
    for k_val in KNN_VALUES:
        splitc_cat: Dict[str, Dict[str, Any]] = {
            f"Split+Clustering [k={k_val}]": _network_clf_models_sign_split_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        m_sc, ps_sc, params_sc = classification_cv_multi(
            dd_exp, splitc_cat, list(dd_exp.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_sc)
        all_params.extend(params_sc)
        _merge_clf_pred_store(pred_store, ps_sc)

    # ── Learned-weight classifiers (SqCorr) ──────────────────────────────────
    print("\nRunning learned-weight classifiers (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lw_cat: Dict[str, Dict[str, Any]] = {
            f"LearnedWeight [k={k_val}]": _learned_weight_clf_models(k_val)
        }
        dd_sq = data_dicts_net[k_val]
        m_lw, ps_lw, params_lw = classification_cv_multi(
            dd_sq, lw_cat, list(dd_sq.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lw)
        all_params.extend(params_lw)
        _merge_clf_pred_store(pred_store, ps_lw)

    # ── Learned-weight classifiers (PCorr) ───────────────────────────────────
    print("\nRunning learned-weight classifiers (PCorr, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lw_pc_cat: Dict[str, Dict[str, Any]] = {
            f"PCorr LearnedWeight [k={k_val}]": _learned_weight_clf_models(k_val)
        }
        dd_pc = data_dicts_pcorr[k_val]
        m_lwp, ps_lwp, params_lwp = classification_cv_multi(
            dd_pc, lw_pc_cat, list(dd_pc.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwp)
        all_params.extend(params_lwp)
        _merge_clf_pred_store(pred_store, ps_lwp)

    # ── Learned-weight + clustering (SqCorr) ─────────────────────────────────
    print("\nRunning learned-weight + clustering classifiers (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lwc_cat: Dict[str, Dict[str, Any]] = {
            f"LW+Clustering [k={k_val}]": _learned_weight_clf_clustering_models(k_val)
        }
        dd_sq = data_dicts_net[k_val]
        m_lwc, ps_lwc, params_lwc = classification_cv_multi(
            dd_sq, lwc_cat, list(dd_sq.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwc)
        all_params.extend(params_lwc)
        _merge_clf_pred_store(pred_store, ps_lwc)

    # ── Learned-weight + clustering (PCorr) ──────────────────────────────────
    print("\nRunning learned-weight + clustering classifiers (PCorr, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lwc_pc_cat: Dict[str, Dict[str, Any]] = {
            f"PCorr LW+Clustering [k={k_val}]": _learned_weight_clf_clustering_models(k_val)
        }
        dd_pc = data_dicts_pcorr[k_val]
        m_lwcp, ps_lwcp, params_lwcp = classification_cv_multi(
            dd_pc, lwc_pc_cat, list(dd_pc.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwcp)
        all_params.extend(params_lwcp)
        _merge_clf_pred_store(pred_store, ps_lwcp)

    # ── PCorr sign-split classifiers ─────────────────────────────────────────
    print("\nRunning PCorr sign-split classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        ps_cat: Dict[str, Dict[str, Any]] = {
            f"PCorr Split [k={k_val}]": _network_clf_models_sign_split()
        }
        dd_pc = data_dicts_pcorr[k_val]
        m_ps, ps_ps, params_ps = classification_cv_multi(
            dd_pc, ps_cat, list(dd_pc.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_ps)
        all_params.extend(params_ps)
        _merge_clf_pred_store(pred_store, ps_ps)

    # ── ExpKernel sign-split classifiers ─────────────────────────────────────
    print("\nRunning ExpKernel sign-split classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        es_cat: Dict[str, Dict[str, Any]] = {
            f"ExpKernel Split [k={k_val}]": _network_clf_models_sign_split()
        }
        dd_exp = data_dicts_exp[k_val]
        m_es, ps_es, params_es = classification_cv_multi(
            dd_exp, es_cat, list(dd_exp.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_es)
        all_params.extend(params_es)
        _merge_clf_pred_store(pred_store, ps_es)

    # ── MI sign-split classifiers ────────────────────────────────────────────
    print("\nRunning MI sign-split classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        ms_cat: Dict[str, Dict[str, Any]] = {
            f"MI Split [k={k_val}]": _network_clf_models_sign_split()
        }
        dd_mi = data_dicts_mi[k_val]
        m_ms, ps_ms, params_ms = classification_cv_multi(
            dd_mi, ms_cat, list(dd_mi.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_ms)
        all_params.extend(params_ms)
        _merge_clf_pred_store(pred_store, ps_ms)

    # ── MI learned-weight classifiers ────────────────────────────────────────
    print("\nRunning MI learned-weight classifiers (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lw_mi_cat: Dict[str, Dict[str, Any]] = {
            f"MI LearnedWeight [k={k_val}]": _learned_weight_clf_models(k_val)
        }
        dd_mi = data_dicts_mi[k_val]
        m_lwm, ps_lwm, params_lwm = classification_cv_multi(
            dd_mi, lw_mi_cat, list(dd_mi.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwm)
        all_params.extend(params_lwm)
        _merge_clf_pred_store(pred_store, ps_lwm)

    # ── MI learned-weight + clustering classifiers ───────────────────────────
    print("\nRunning MI learned-weight + clustering classifiers (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 -> no valid m
        lwc_mi_cat: Dict[str, Dict[str, Any]] = {
            f"MI LW+Clustering [k={k_val}]": _learned_weight_clf_clustering_models(k_val)
        }
        dd_mi = data_dicts_mi[k_val]
        m_lwcm, ps_lwcm, params_lwcm = classification_cv_multi(
            dd_mi, lwc_mi_cat, list(dd_mi.keys()),
            n_splits=n_splits,
            sample_tickers=SAMPLE_TICKERS,
            spike_quantile=INDEX_SPIKE_QUANTILE,
            spike_lookback=INDEX_SPIKE_LOOKBACK,
            save_params=True, num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwcm)
        all_params.extend(params_lwcm)
        _merge_clf_pred_store(pred_store, ps_lwcm)

    # ── Aggregate and save ───────────────────────────────────────────────────
    metrics_df = pd.concat([metrics_df] + all_extra_metrics, ignore_index=True)

    summary_raw = summarize_classification(metrics_df)
    save_classification_results(metrics_df, summary_raw, RESULTS_DIR)
    save_model_params(all_params, RESULTS_DIR, "classification_model_params.json")
    save_classification_prediction_store(pred_store, RESULTS_DIR)

    # Coalesce k-variants for cleaner reporting
    metrics_coalesced = coalesce_categories(
        metrics_df, metric_col="ROC_AUC", higher_is_better=True,
    )
    summary = summarize_classification(metrics_coalesced)

    # ── Validation / Test reporting ──────────────────────────────────────────
    if n_splits > 1:
        print("\n" + "=" * 80)
        print("VALIDATION-FOLD SUMMARY  (Fold 1 — used for model selection)")
        print("=" * 80)
        val_metrics = metrics_coalesced[metrics_coalesced["Fold"] == 1]
        val_summary = summarize_classification(val_metrics)
        print_classification_summary(val_summary, title="Validation Summary (all models)")
        print_best_classifier(val_summary)

        print("\n" + "=" * 80)
        print("TEST-FOLD SUMMARY  (Fold 0 — held-out final evaluation)")
        print("=" * 80)
        test_metrics = metrics_coalesced[metrics_coalesced["Fold"] == 0]
        test_summary = summarize_classification(test_metrics)
        print_classification_summary(test_summary, title="Test Summary (all models)")
        print_best_classifier(test_summary)

        best_models = select_best_on_validation(metrics_coalesced, val_fold=1, metric="ROC_AUC")
        print("\n" + "=" * 80)
        print("BEST CLASSIFIER PER CATEGORY  (chosen on validation fold)")
        print("=" * 80)
        print(best_models.to_string(index=False))
        print_best_clf_test_summary(metrics_coalesced, best_models, test_fold=0)
        summary_title = "Classification Summary (all indices)"
    else:
        print("\n" + "=" * 80)
        print("TEST SUMMARY  (Fold 0 — held-out final evaluation)")
        print("=" * 80)
        test_metrics = metrics_coalesced[metrics_coalesced["Fold"] == 0]
        test_summary = summarize_classification(test_metrics)
        print_classification_summary(test_summary, title="Test Summary (all models)")
        print_best_classifier(test_summary)
        summary_title = "Classification Summary — single hold-out fold (all indices)"

    print_classification_summary(summary, title=summary_title)
    print_best_classifier(summary)
    print_compact_clf_leaderboard(summary)
    print_overall_best(summary)
    print_wilcoxon_best_network_vs_baseline_clf(metrics_coalesced)
    print_per_ticker_network_vs_baseline_clf(metrics_coalesced, SAMPLE_TICKERS)
    print_per_ticker_classification(metrics_coalesced, SAMPLE_TICKERS)
    print_k_breakdown_clf(metrics_df)
    print_weighting_scheme_breakdown_clf(metrics_coalesced)


def sanity() -> None:
    """
    Fast smoke-test entry point for the index classification pipeline.

    Runs one model per baseline family + two network models (NetHAR, NetVAR)
    on a single SqCorr graph at k=3, using all 21 index tickers.  Intended
    to verify the full pipeline works without committing to a multi-hour run.

    To restore the full experiment, replace the ``sanity()`` call in the
    ``__main__`` block below with ``main()``.
    """
    SAMPLE_TICKERS = ["SPX2", "FTSE2", "N2252", "GDAXI2", "IXIC2"]
    RESULTS_DIR = Path(__file__).parent.parent / "results" / "index_results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    K_VAL = 3
    INDEX_SPIKE_QUANTILE = 0.75
    INDEX_SPIKE_LOOKBACK = 252 * 3

    graph_n_jobs = max(1, min(8, (os.cpu_count() or 1) - 1))
    cv_n_jobs = max(1, min(24, (os.cpu_count() or 1) - 1))

    print("Loading and preprocessing index data...")
    data_dict = get_index_data_for_har()
    tickers = list(data_dict.keys())
    print(f"  {len(tickers)} indices — sanity mode (1 model/family, SqCorr k={K_VAL} only)")

    # ── Graph: reuse cached SqCorr from index regression run if available ────
    from data.graph_cache import load_graph_data, graph_cache_exists
    CACHE_DIR = RESULTS_DIR / "graph_cache"
    cache_loaded = False
    if graph_cache_exists(CACHE_DIR, "sqcorr", [K_VAL]):
        print(f"\nLoading cached SqCorr graph features (k={K_VAL})...")
        data_dicts_net, _ = load_graph_data(CACHE_DIR, "sqcorr", [K_VAL])
        cache_loaded = True
        print("  Graph cache loaded.")
    else:
        print(f"\nBuilding SqCorr graph (k={K_VAL}) from scratch...")
        net = SquaredCorrelationNetwork(
            window=60, step=1, save_step=5, n_jobs=graph_n_jobs,
            graph_type="knn", k=K_VAL,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_net = {K_VAL: net.fit_transform(data_dict)}
        print(f"  [SqCorr] k={K_VAL}: {net.n_all_snapshots_} total, {net.n_snapshots_} saved.")

    # ── Slim baseline catalogue: 1 entry per family ──────────────────────────
    baseline_catalogue: Dict[str, Dict[str, Any]] = {
        "HAR-Logit": {
            "HAR-Logit (C=1.0)":      (HARLogitClassifier(C=1.0, use_market=False),              False),
        },
        "HAR-Ext-Logit": {
            "HAR-Ext-Logit (C=1.0)":  (HARExtendedLogitClassifier(C=1.0, use_market=False),      False),
        },
        "RegimeSwitching-Logit": {
            "RegHAR-Logit (p50)":     (RegimeSwitchingHARLogitClassifier(regime_percentile=0.5,
                                                                          use_market=False),       False),
        },
        "DCC-GARCH-Logit": {
            "DCC-GARCH-Logit (C=1.0)":(DCCGARCHSpikeClassifier(C=1.0, aux_returns_col=None,
                                                                 returns_multiplier=100.0),       False),
        },
    }

    print(f"\nRunning baseline classifiers on {len(tickers)} indices...")
    metrics_df, pred_store, all_params = classification_cv_multi(
        data_dict, baseline_catalogue, tickers,
        n_splits=1,
        sample_tickers=SAMPLE_TICKERS,
        spike_quantile=INDEX_SPIKE_QUANTILE,
        spike_lookback=INDEX_SPIKE_LOOKBACK,
        save_params=True,
        num_workers=cv_n_jobs,
    )

    # ── Single SqCorr network run at k=3 ─────────────────────────────────────
    net_catalogue: Dict[str, Dict[str, Any]] = {
        f"Network [k={K_VAL}]": {
            "NetHAR-Logit (C=1.0)":         (NetworkHARClassifier(C=1.0, use_market=False),     False),
            "NetVAR-Logit (C=1,a=0.1,b=1)": (NetworkVARClassifier(C_stage1=1.0, stage2_alpha=0.1,
                                                                    correction_bound=1.0,
                                                                    use_market=False),            False),
        },
    }
    dd_net = data_dicts_net[K_VAL]
    print(f"\nRunning network classifiers (SqCorr, k={K_VAL})...")
    m_net, ps_net, params_net = classification_cv_multi(
        dd_net, net_catalogue, list(dd_net.keys()),
        n_splits=1,
        sample_tickers=SAMPLE_TICKERS,
        spike_quantile=INDEX_SPIKE_QUANTILE,
        spike_lookback=INDEX_SPIKE_LOOKBACK,
        save_params=True,
        num_workers=cv_n_jobs,
    )
    _merge_clf_pred_store(pred_store, ps_net)
    all_params.extend(params_net)

    # ── Aggregate and save ────────────────────────────────────────────────────
    metrics_df = pd.concat([metrics_df, m_net], ignore_index=True)
    summary = summarize_classification(metrics_df)
    save_classification_results(metrics_df, summary, RESULTS_DIR)
    save_model_params(all_params, RESULTS_DIR, "classification_model_params.json")
    save_classification_prediction_store(pred_store, RESULTS_DIR)

    print_classification_summary(summary, title="Index Classification (sanity — 1 model/family, SqCorr k=3)")
    print_best_classifier(summary)
    print_compact_clf_leaderboard(summary)


if __name__ == "__main__":
    # sanity()
    main()
