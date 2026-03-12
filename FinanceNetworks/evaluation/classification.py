"""
evaluation/classification.py
=============================
Expanding-window cross-validation for volatility-spike classifiers.

Mirrors cross_val.py but:
  - Uses binary spike labels computed fresh from training-fold Y_fwd on
    every fold (no look-ahead leakage).
  - Evaluates with Accuracy, Precision, Recall, and ROC-AUC.
  - Selects the best model by mean ROC-AUC across all tickers.
  - Saves fold-level results to classification_results.json and
    aggregated summary to classification_summary.json.

Entry points
------------
run_classification_cv         : Core CV loop (returns metrics_df, pred_store).
save_classification_results   : Persist outputs to disk.
main()                        : Full run on all tickers.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from data.volatility_spikes import compute_spike_threshold, add_spike_label
from data.preprocess import remove_outliers

# Reuse the expanding-window fold generator from cross_val
from evaluation.cross_val import expanding_folds
from evaluation.interpretability import extract_model_params, save_model_params, save_graph_snapshots

from visualize.print_results_classification import (
    summarize_classification,
    print_classification_summary,
    print_best_classifier,
    print_overall_best,
    print_per_ticker_classification,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def weighted_recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_rv: np.ndarray,
) -> float:
    """
    Weighted recall: fraction of spike *mass* correctly identified.

    Each true spike day is weighted by log(1 + Y_fwd), so that missing a
    large-volatility event is penalised more than missing a small one.

        weighted_recall = Σ w_t · 1[label=1, pred=1]
                        / Σ w_t · 1[label=1]

    where w_t = log(1 + Y_fwd_t).

    Returns NaN when there are no positive labels in the fold (undefined).
    """
    mask_pos = y_true == 1
    if not mask_pos.any():
        return float("nan")
    w = np.log1p(y_rv)           # log(1 + Y_fwd), element-wise
    denom = w[mask_pos].sum()
    if denom == 0:
        return float("nan")
    numer = w[mask_pos & (y_pred == 1)].sum()
    return float(numer / denom)


def eval_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    y_rv: np.ndarray,
) -> dict:
    """
    Compute F1, Precision, Recall, ROC-AUC, Weighted Recall, and Accuracy.

    Parameters
    ----------
    y_true  : True binary labels (0 / 1).
    y_pred  : Hard class predictions (0 / 1).
    y_proba : Predicted probability of class 1 (used for ROC-AUC).
    y_rv    : Raw forward-RV values (Y_fwd) for the test rows, used to
              weight recall by spike magnitude.

    Returns
    -------
    Dictionary with keys F1, Precision, Recall, ROC_AUC,
    Weighted_Recall, Accuracy.

    Notes
    -----
    ROC-AUC and Weighted_Recall are set to NaN for degenerate folds
    (only one class present); NaNs are ignored during aggregation.
    Precision, Recall, and F1 use zero_division=0 so that folds where
    the classifier predicts no positive class don't raise exceptions.
    Primary evaluation metric is F1 (balances precision and recall,
    robust to class imbalance unlike accuracy).
    """
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")

    return {
        "F1":              float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision":       float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":          float(recall_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC":         auc,
        "Weighted_Recall": weighted_recall_score(y_true, y_pred, y_rv),
        "Accuracy":        float(accuracy_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Fit / predict helper
# ---------------------------------------------------------------------------

def fit_predict_classifier(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit *model* on (X_train, y_train) and return predictions for X_test.

    ``model.features`` must list the column names needed by the model.

    Returns
    -------
    y_pred  : Hard predictions (0 / 1) for X_test.
    y_proba : Probability of class 1 for X_test.
    """
    model.fit(X_train[model.features], y_train)
    y_pred  = model.predict(X_test[model.features])
    y_proba = model.predict_proba(X_test[model.features])[:, 1]
    return y_pred, y_proba


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------

def run_classification_cv(
    data_dict: Dict[str, pd.DataFrame],
    model_catalogue: Dict[str, Dict[str, Any]],
    tickers: List[str],
    n_splits: int = 1,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
    spike_quantile: float = 0.8,
    sample_tickers: Optional[List[str]] = None,
    save_params: bool = False,
) -> "tuple[pd.DataFrame, dict, list] | tuple[pd.DataFrame, dict]":
    """
    Expanding-window cross-validation for volatility-spike classifiers.

    Spike-label construction (no leakage)
    --------------------------------------
    On every fold the spike threshold is computed **only** from the training
    slice of Y_fwd.  The same scalar threshold is then applied to convert
    Y_fwd → binary labels for both the training and test portion of that fold.
    Test-period statistics are never used to determine "what counts as a spike".

    Parameters
    ----------
    data_dict       : {ticker: DataFrame} with HAR features (log_RV1 etc.) and
                      Y_fwd column (produced by preprocess_for_har or
                      preprocess_for_classification).
    model_catalogue : Nested dict {category: {model_name: model_entry}}
                      where model_entry is either a model instance or a
                      (model_instance, remove_outliers_bool) tuple.
    n_splits        : Number of expanding-window folds.
    test_size       : Number of observations per test fold (default ≈ 1 year).
    min_train_size  : Minimum training observations required.
    spike_quantile  : Quantile of training Y_fwd above which a day is a spike
                      (default 0.8 → top-20 %).
    sample_tickers  : Store per-row predictions for these tickers.

    Returns
    -------
    metrics_df : Long-form DataFrame with columns
                 [Category, Ticker, Model, Fold,
                  Accuracy, Precision, Recall, ROC_AUC].
    pred_store : {ticker: DataFrame} with predicted labels and spike
                 probabilities for sample tickers.
    """
    rows: list = []
    pred_store: dict = {}
    params_store: list = []
    sample_set = set(sample_tickers or [])

    for t in tqdm(tickers, desc="Tickers", unit="ticker"):
        if t not in data_dict:
            continue

        feat = data_dict[t]
        if len(feat) < min_train_size + test_size:
            continue

        if t in sample_set:
            pred_df = pd.DataFrame(index=feat.index)

        for fold_id, (train_idx, test_idx) in enumerate(
            expanding_folds(feat.index, n_splits, test_size, min_train_size)
        ):
            X_train = feat.loc[train_idx]
            X_test  = feat.loc[test_idx]

            # ── Spike labels (threshold from training fold only) ────────────
            threshold = compute_spike_threshold(
                X_train["Y_fwd"], quantile=spike_quantile
            )
            y_train = add_spike_label(X_train, threshold)["spike"]
            y_test  = add_spike_label(X_test,  threshold)["spike"]

            for category, models in model_catalogue.items():
                for model_name, model_entry in models.items():
                    if isinstance(model_entry, tuple):
                        model_template, do_remove_outliers = model_entry
                    else:
                        model_template, do_remove_outliers = model_entry, False

                    model = copy.deepcopy(model_template)

                    # Optionally strip outlier rows from training features
                    X_train_fit = (
                        remove_outliers(X_train) if do_remove_outliers else X_train
                    )
                    # Realign y_train to the (possibly reduced) training index
                    y_train_fit = y_train.loc[X_train_fit.index]

                    try:
                        y_pred, y_proba = fit_predict_classifier(
                            model, X_train_fit, X_test, y_train_fit
                        )
                    except Exception as exc:
                        print(
                            f"  [SKIP] {t} | fold {fold_id} | "
                            f"{category}/{model_name}: {exc}"
                        )
                        continue

                    m = eval_classification(y_test.values, y_pred, y_proba,
                                            y_rv=X_test["Y_fwd"].values)
                    rows.append(
                        {
                            "Category": category,
                            "Ticker":   t,
                            "Model":    model_name,
                            "Fold":     fold_id,
                            **m,
                        }
                    )

                    # ── Save fitted parameters for interpretability ────────
                    if save_params:
                        try:
                            p = extract_model_params(model)
                            p["category"]    = category
                            p["model_name"]  = model_name
                            p["ticker"]      = t
                            p["fold"]        = fold_id
                            p["train_start"] = str(train_idx[0])
                            p["train_end"]   = str(train_idx[-1])
                            p["test_start"]  = str(test_idx[0])
                            p["test_end"]    = str(test_idx[-1])
                            params_store.append(p)
                        except Exception:
                            pass  # don't break CV for param extraction failures

                    if t in sample_set:
                        col_key = f"[{category}] {model_name}"
                        pred_df.loc[test_idx, f"{col_key}_pred"]  = y_pred
                        pred_df.loc[test_idx, f"{col_key}_proba"] = y_proba

        if t in sample_set:
            # Attach ground-truth spike labels (global threshold for reference)
            global_thr = compute_spike_threshold(
                feat["Y_fwd"], quantile=spike_quantile
            )
            pred_df["Y_true_spike"] = (feat["Y_fwd"] > global_thr).astype(int)
            pred_store[t] = pred_df

    metrics_df = pd.DataFrame(rows)
    if save_params:
        return metrics_df, pred_store, params_store
    return metrics_df, pred_store


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_classification_results(
    metrics_df: pd.DataFrame,
    summary: pd.DataFrame,
    results_dir: Path,
) -> None:
    """
    Persist cross-validation outputs to disk.

    Files written
    -------------
    classification_results.json : Fold-level metrics, one record per
                                  (Category, Ticker, Model, Fold).
    classification_summary.json : Model-level summary aggregated across
                                  tickers, sorted by mean_ROC_AUC descending.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / "classification_results.json"
    metrics_df.to_json(raw_path, orient="records", indent=2)
    print(f"Fold-level metrics  -> {raw_path}")

    summary_path = results_dir / "classification_summary.json"
    summary.reset_index().to_json(summary_path, orient="records", indent=2)
    print(f"Summary metrics     -> {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data import get_data_for_har
    from models.baselines_classification import (
        HARLogitClassifier,
        HARExtendedLogitClassifier,
        RegimeSwitchingHARLogitClassifier,
    )
    from models.network_models_classification import (
        NetworkHARClassifier,
        NetworkVARClassifier,
    )
    from models.correlation_network import SquaredCorrelationNetwork, PartialCorrelationNetwork, MutualInformationNetwork

    SAMPLE_TICKERS = ["AAPL", "TSLA", "GOOG", "META", "MSFT", "NVDA", "NFLX", "AMZN"]
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    KNN_VALUES = [1, 3, 5]

    print("Loading and preprocessing data...")
    data_dict = get_data_for_har(100)
    tickers = list(data_dict.keys())

    # ── Offline graph builds (identical pattern to cross_val.py main()) ──────
    print("\nBuilding squared-correlation networks (k=1, 3, 5)...")
    data_dicts_net: dict = {}
    for k_val in KNN_VALUES:
        net_k = SquaredCorrelationNetwork(
            window=60, step=5, graph_type="knn", k=k_val,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_net[k_val] = net_k.fit_transform(data_dict)
        print(f"  [SqCorr] k={k_val}: built {net_k.n_snapshots_} graph snapshots.")

    print("\nBuilding partial-correlation networks (k=1, 3, 5)...")
    data_dicts_pcorr: dict = {}
    for k_val in KNN_VALUES:
        net_pk = PartialCorrelationNetwork(
            window=60, step=5, graph_type="knn", k=k_val, shrinkage=0.1,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_pcorr[k_val] = net_pk.fit_transform(data_dict)
        print(f"  [PCorr]  k={k_val}: built {net_pk.n_snapshots_} snapshots.")

    print("\nBuilding exp-kernel networks (k=1, 3, 5)...")
    data_dicts_exp: dict = {}
    for k_val in KNN_VALUES:
        net_exp = SquaredCorrelationNetwork(
            window=60, step=5, graph_type="knn", k=k_val,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            idw_kernel="exp", exp_lambda=5.0,
        )
        data_dicts_exp[k_val] = net_exp.fit_transform(data_dict)
        print(f"  [ExpKernel] k={k_val}: built {net_exp.n_snapshots_} snapshots.")

    print("\nBuilding mutual-information networks (k=1, 3, 5)...")
    data_dicts_mi: dict = {}
    for k_val in KNN_VALUES:
        net_mi = MutualInformationNetwork(
            window=60, step=5, graph_type="knn", k=k_val,
            n_bins=10,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_mi[k_val] = net_mi.fit_transform(data_dict)
        print(f"  [MI]     k={k_val}: built {net_mi.n_snapshots_} snapshots.")

    # ── Model catalogues ─────────────────────────────────────────────────────

    baseline_catalogue: Dict[str, Dict[str, Any]] = {
        "HAR-Logit": {
            "HAR-Logit (C=0.01)":        (HARLogitClassifier(C=0.01),             False),
            "HAR-Logit (C=0.1)":         (HARLogitClassifier(C=0.1),              False),
            "HAR-Logit (C=1.0)":         (HARLogitClassifier(C=1.0),              False),
            "HAR-Logit (C=10.0)":        (HARLogitClassifier(C=10.0),             False),
            "HAR-Logit (C=1.0, no-out)": (HARLogitClassifier(C=1.0),              True),
        },
        "HAR-Ext-Logit": {
            "HAR-Ext-Logit (C=0.01)":        (HARExtendedLogitClassifier(C=0.01),  False),
            "HAR-Ext-Logit (C=0.1)":         (HARExtendedLogitClassifier(C=0.1),   False),
            "HAR-Ext-Logit (C=1.0)":         (HARExtendedLogitClassifier(C=1.0),   False),
            "HAR-Ext-Logit (C=10.0)":        (HARExtendedLogitClassifier(C=10.0),  False),
            "HAR-Ext-Logit (C=1.0, no-out)": (HARExtendedLogitClassifier(C=1.0),   True),
        },
        "RegimeSwitching-Logit": {
            "RegHAR-Logit (p50)":            (RegimeSwitchingHARLogitClassifier(regime_percentile=0.5),  False),
            "RegHAR-Logit (p75)":            (RegimeSwitchingHARLogitClassifier(regime_percentile=0.75), False),
            "RegHAR-Logit (p50, C=0.1)":     (RegimeSwitchingHARLogitClassifier(C=0.1, regime_percentile=0.5), False),
        },
    }

    def _network_clf_models() -> Dict[str, Any]:
        """Fresh network classifier instances (needed per k-value run)."""
        return {
            "NetHAR-Logit (C=0.1)":          (NetworkHARClassifier(C=0.1),             False),
            "NetHAR-Logit (C=1.0)":          (NetworkHARClassifier(C=1.0),             False),
            "NetHAR-Logit (C=10.0)":         (NetworkHARClassifier(C=10.0),            False),
            "NetVAR-Logit (C=1,a=0.1,b=1)":  (NetworkVARClassifier(C_stage1=1.0,
                                                                     stage2_alpha=0.1,
                                                                     correction_bound=1.0), False),
            "NetVAR-Logit (C=1,a=0.5,b=2)":  (NetworkVARClassifier(C_stage1=1.0,
                                                                     stage2_alpha=0.5,
                                                                     correction_bound=2.0), False),
            "NetVAR-Logit (C=1,a=1.0,b=2)":  (NetworkVARClassifier(C_stage1=1.0,
                                                                     stage2_alpha=1.0,
                                                                     correction_bound=2.0), False),
        }

    def _network_clf_models_clustering() -> Dict[str, Any]:
        """Network classifier models with clustering features enabled."""
        return {
            "NetHAR+C-Logit (C=0.1)":         (NetworkHARClassifier(C=0.1,  use_clustering=True), False),
            "NetHAR+C-Logit (C=1.0)":         (NetworkHARClassifier(C=1.0,  use_clustering=True), False),
            "NetVAR+C-Logit (C=1,a=0.1,b=1)": (NetworkVARClassifier(C_stage1=1.0,
                                                                      stage2_alpha=0.1,
                                                                      correction_bound=1.0,
                                                                      use_clustering=True), False),
            "NetVAR+C-Logit (C=1,a=0.5,b=2)": (NetworkVARClassifier(C_stage1=1.0,
                                                                      stage2_alpha=0.5,
                                                                      correction_bound=2.0,
                                                                      use_clustering=True), False),
        }

    # ── Run HAR-Logit baselines ──────────────────────────────────────────────
    print(f"\nRunning baseline classifiers on {len(tickers)} tickers...")
    metrics_df, pred_store, all_params = run_classification_cv(
        data_dict,
        baseline_catalogue,
        tickers,
        n_splits=1,
        sample_tickers=SAMPLE_TICKERS,
        spike_quantile=0.8,
        save_params=True,
    )
    all_extra_metrics: list = []

    # Save graph snapshots for interpretability
    print("\nSaving graph snapshots...")
    GRAPHS_DIR = RESULTS_DIR / "graphs"
    for k_val in KNN_VALUES:
        # Graphs are the same objects built above; save once
        pass  # (saved by cross_val.py main; avoid duplicating if run independently)

    # ── Squared-correlation network classifiers ───────────────────────────────
    print("\nRunning network classifiers (SqCorr, k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        net_cat: Dict[str, Dict[str, Any]] = {
            f"Network [k={k_val}]": _network_clf_models()
        }
        dd_net = data_dicts_net[k_val]
        m_k, ps_k, params_k = run_classification_cv(
            dd_net, net_cat, list(dd_net.keys()),
            n_splits=1, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
        )
        all_extra_metrics.append(m_k)
        all_params.extend(params_k)
        for t in ps_k:
            new_cols = [c for c in ps_k[t].columns if c not in ("Y_true_spike",)]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_k[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_k[t]

    # ── Partial-correlation network classifiers ───────────────────────────────
    print("\nRunning network classifiers (PCorr, k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        pc_cat: Dict[str, Dict[str, Any]] = {
            f"PCorr Network [k={k_val}]": _network_clf_models()
        }
        dd_pc = data_dicts_pcorr[k_val]
        m_pk, ps_pk, params_pk = run_classification_cv(
            dd_pc, pc_cat, list(dd_pc.keys()),
            n_splits=1, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
        )
        all_extra_metrics.append(m_pk)
        all_params.extend(params_pk)
        for t in ps_pk:
            new_cols = [c for c in ps_pk[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_pk[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_pk[t]

    # ── Exp-kernel network classifiers ────────────────────────────────────────
    print("\nRunning network classifiers (ExpKernel, k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        exp_cat: Dict[str, Dict[str, Any]] = {
            f"ExpKernel [k={k_val}]": _network_clf_models()
        }
        dd_exp = data_dicts_exp[k_val]
        m_ek, ps_ek, params_ek = run_classification_cv(
            dd_exp, exp_cat, list(dd_exp.keys()),
            n_splits=1, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
        )
        all_extra_metrics.append(m_ek)
        all_params.extend(params_ek)
        for t in ps_ek:
            new_cols = [c for c in ps_ek[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_ek[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_ek[t]

    # ── SqCorr + clustering classifiers ──────────────────────────────────────
    print("\nRunning clustering-feature classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        cl_cat: Dict[str, Dict[str, Any]] = {
            f"Clustering [k={k_val}]": _network_clf_models_clustering()
        }
        dd_sq = data_dicts_net[k_val]
        m_cl, ps_cl, params_cl = run_classification_cv(
            dd_sq, cl_cat, list(dd_sq.keys()),
            n_splits=1, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
        )
        all_extra_metrics.append(m_cl)
        all_params.extend(params_cl)
        for t in ps_cl:
            new_cols = [c for c in ps_cl[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_cl[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_cl[t]

    # ── Exp + clustering classifiers ──────────────────────────────────────────
    print("\nRunning exp-kernel + clustering classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        ec_cat: Dict[str, Dict[str, Any]] = {
            f"Exp+Clustering [k={k_val}]": _network_clf_models_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        m_ec, ps_ec, params_ec = run_classification_cv(
            dd_exp, ec_cat, list(dd_exp.keys()),
            n_splits=1, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
        )
        all_extra_metrics.append(m_ec)
        all_params.extend(params_ec)
        for t in ps_ec:
            new_cols = [c for c in ps_ec[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_ec[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_ec[t]

    # ── Mutual-information network classifiers ────────────────────────────────
    print("\nRunning mutual-information network classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        mi_cat: Dict[str, Dict[str, Any]] = {
            f"MI Network [k={k_val}]": _network_clf_models()
        }
        dd_mi = data_dicts_mi[k_val]
        m_mi, ps_mi, params_mi = run_classification_cv(
            dd_mi, mi_cat, list(dd_mi.keys()),
            n_splits=1, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
        )
        all_extra_metrics.append(m_mi)
        all_params.extend(params_mi)
        for t in ps_mi:
            new_cols = [c for c in ps_mi[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_mi[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_mi[t]

    # ── Merge all results ─────────────────────────────────────────────────────
    metrics_df = pd.concat([metrics_df] + all_extra_metrics, ignore_index=True)

    summary = summarize_classification(metrics_df)
    save_classification_results(metrics_df, summary, RESULTS_DIR)
    save_model_params(all_params, RESULTS_DIR, "classification_model_params.json")

    print_classification_summary(summary, title="Classification Summary (all stocks)")
    print_best_classifier(summary)
    print_overall_best(summary)
    print_per_ticker_classification(metrics_df, SAMPLE_TICKERS)


if __name__ == "__main__":
    main()
