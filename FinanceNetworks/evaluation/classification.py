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

import argparse
import copy
import multiprocessing as mp
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
from evaluation.cross_val import expanding_folds, _with_no_outlier_variants, select_best_on_validation
from evaluation.interpretability import (
    extract_model_params,
    save_model_params,
    save_graph_snapshots,
    save_feature_snapshots,
)

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
    load_and_print_classification_results,
)
from visualize.utils import coalesce_categories

# Module-level data store.  Populated in classification_cv_multi() before
# workers are forked so that DataFrames are inherited copy-on-write.
_CLF_CV_DATA: dict = {}


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
    # sklearn may emit UndefinedMetricWarning rather than raising when the
    # evaluation fold contains only one class. Treat these folds as undefined
    # for ROC-AUC and let downstream aggregation ignore the NaN.
    if np.unique(y_true).size < 2:
        auc = float("nan")
    else:
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
    spike_lookback: Optional[int] = None,
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
    spike_lookback  : Optional recent-window length used when computing the
                      training-fold spike threshold. When None, the full
                      training fold is used.
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
                X_train["Y_fwd"], quantile=spike_quantile, lookback=spike_lookback
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
                feat["Y_fwd"], quantile=spike_quantile, lookback=spike_lookback
            )
            pred_df["Y_true_spike"] = (feat["Y_fwd"] > global_thr).astype(int)
            pred_store[t] = pred_df

    metrics_df = pd.DataFrame(rows)
    if save_params:
        return metrics_df, pred_store, params_store
    return metrics_df, pred_store


def _run_single_clf_task(task: "dict[str, Any]") -> "dict[str, Any]":
    """Execute one (ticker, fold, model) classification task."""
    feat = _CLF_CV_DATA[task["ticker"]]
    train_idx = task["train_idx"]
    test_idx = task["test_idx"]

    X_train = feat.loc[train_idx]
    X_test = feat.loc[test_idx]

    # Spike labels (threshold from training fold only)
    threshold = compute_spike_threshold(
        X_train["Y_fwd"],
        quantile=task["spike_quantile"],
        lookback=task["spike_lookback"],
    )
    y_train = add_spike_label(X_train, threshold)["spike"]
    y_test = add_spike_label(X_test, threshold)["spike"]

    model = copy.deepcopy(task["model_template"])
    X_train_fit = remove_outliers(X_train) if task["do_remove_outliers"] else X_train
    y_train_fit = y_train.loc[X_train_fit.index]

    try:
        y_pred, y_proba = fit_predict_classifier(
            model, X_train_fit, X_test, y_train_fit
        )
    except Exception as exc:
        return {
            "row": None,
            "prediction": None,
            "params": None,
            "error": (
                f"{task['ticker']} | fold {task['fold_id']} | "
                f"{task['category']}/{task['model_name']}: {exc}"
            ),
        }

    m = eval_classification(
        y_test.values, y_pred, y_proba, y_rv=X_test["Y_fwd"].values
    )

    result: dict = {
        "row": {
            "Category": task["category"],
            "Ticker": task["ticker"],
            "Model": task["model_name"],
            "Fold": task["fold_id"],
            **m,
        },
        "prediction": None,
        "params": None,
        "error": None,
    }

    if task["store_predictions"]:
        result["prediction"] = {
            "ticker": task["ticker"],
            "col_key": f"[{task['category']}] {task['model_name']}",
            "test_idx": test_idx,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    if task["save_params"]:
        try:
            params = extract_model_params(model)
            params["category"] = task["category"]
            params["model_name"] = task["model_name"]
            params["ticker"] = task["ticker"]
            params["fold"] = task["fold_id"]
            params["train_start"] = str(train_idx[0])
            params["train_end"] = str(train_idx[-1])
            params["test_start"] = str(test_idx[0])
            params["test_end"] = str(test_idx[-1])
            result["params"] = params
        except Exception:
            pass

    return result


def classification_cv_multi(
    data_dict: Dict[str, pd.DataFrame],
    model_catalogue: Dict[str, Dict[str, Any]],
    tickers: List[str],
    n_splits: int = 1,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
    spike_quantile: float = 0.8,
    spike_lookback: Optional[int] = None,
    sample_tickers: Optional[List[str]] = None,
    save_params: bool = False,
    num_workers: int = 24,
) -> "tuple[pd.DataFrame, dict, list] | tuple[pd.DataFrame, dict]":
    """
    Fork-based multiprocessing CV for classification.

    Mirrors run_classification_cv but distributes (ticker, fold, model) tasks
    across worker processes for significant speed-up.
    """
    rows: list = []
    pred_store: dict = {}
    params_store: list = []
    sample_set = set(sample_tickers or [])
    tasks: list = []

    for ticker in tickers:
        if ticker not in data_dict:
            continue
        feat = data_dict[ticker]
        if len(feat) < min_train_size + test_size:
            continue

        if ticker in sample_set:
            pred_store[ticker] = pd.DataFrame(index=feat.index)

        for fold_id, (train_idx, test_idx) in enumerate(
            expanding_folds(feat.index, n_splits, test_size, min_train_size)
        ):
            for category, models in model_catalogue.items():
                for model_name, model_entry in models.items():
                    if isinstance(model_entry, tuple):
                        model_template, do_remove_outliers = model_entry
                    else:
                        model_template, do_remove_outliers = model_entry, False

                    tasks.append({
                        "ticker": ticker,
                        "fold_id": fold_id,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "category": category,
                        "model_name": model_name,
                        "model_template": model_template,
                        "do_remove_outliers": do_remove_outliers,
                        "spike_quantile": spike_quantile,
                        "spike_lookback": spike_lookback,
                        "store_predictions": ticker in sample_set,
                        "save_params": save_params,
                    })

    if not tasks:
        metrics_df = pd.DataFrame(rows)
        if save_params:
            return metrics_df, pred_store, params_store
        return metrics_df, pred_store

    global _CLF_CV_DATA
    _CLF_CV_DATA = data_dict

    max_procs = max(1, int(num_workers))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max_procs) as pool:
        for result in tqdm(
            pool.imap_unordered(_run_single_clf_task, tasks),
            total=len(tasks),
            desc="Clf CV tasks",
            unit="task",
        ):
            if result["error"]:
                print(f"  [SKIP] {result['error']}")
                continue

            rows.append(result["row"])

            pred_result = result["prediction"]
            if pred_result is not None:
                t = pred_result["ticker"]
                col_key = pred_result["col_key"]
                pred_store[t].loc[
                    pred_result["test_idx"], f"{col_key}_pred"
                ] = pred_result["y_pred"]
                pred_store[t].loc[
                    pred_result["test_idx"], f"{col_key}_proba"
                ] = pred_result["y_proba"]

            if result["params"] is not None:
                params_store.append(result["params"])

    # Attach ground-truth spike labels for sample tickers
    for t in sample_set:
        if t in pred_store and t in data_dict:
            feat = data_dict[t]
            global_thr = compute_spike_threshold(
                feat["Y_fwd"], quantile=spike_quantile, lookback=spike_lookback
            )
            pred_store[t]["Y_true_spike"] = (
                feat["Y_fwd"] > global_thr
            ).astype(int)

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


def save_classification_prediction_store(
    pred_store: Dict[str, pd.DataFrame],
    results_dir: Path,
    folder_name: str = "predictions_classification",
) -> None:
    """Persist per-ticker classifier predictions/probabilities for plotting."""
    out_dir = Path(results_dir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for ticker, df in pred_store.items():
        if df is None or df.empty:
            continue
        out_path = out_dir / f"{ticker}_classification_predictions.csv"
        df_out = df.copy()
        df_out.index = pd.to_datetime(df_out.index)
        df_out.index.name = "Date"
        df_out.to_csv(out_path)
        n_written += 1

    print(f"Prediction files    -> {out_dir}  ({n_written} tickers)")


def print_best_clf_test_summary(
    metrics_df: pd.DataFrame,
    best_models: pd.DataFrame,
    test_fold: int = 0,
) -> None:
    """
    Print classification summary for only the best models (selected on
    validation fold) evaluated on the test fold.
    """
    test_rows = metrics_df[metrics_df["Fold"] == test_fold]
    best_set = set(zip(best_models["Category"], best_models["Model"]))
    mask = test_rows.apply(lambda r: (r["Category"], r["Model"]) in best_set, axis=1)
    test_best = test_rows[mask].copy()

    if test_best.empty:
        print("No test-fold results for the selected best models.")
        return

    summary = summarize_classification(test_best)
    print("\n" + "=" * 80)
    print("BEST CLASSIFIERS  (selected on validation fold, evaluated on test fold)")
    print("=" * 80)
    print_classification_summary(summary, title="Test-Set Classification Summary (best per category)")
    print_best_classifier(summary)
    print_compact_clf_leaderboard(summary)

    # Also print val-fold metrics of the best models for comparison
    val_rows = metrics_df[metrics_df["Fold"] == 1]
    val_mask = val_rows.apply(lambda r: (r["Category"], r["Model"]) in best_set, axis=1)
    val_best = val_rows[val_mask].copy()
    if not val_best.empty:
        val_summary = summarize_classification(val_best)
        print_classification_summary(val_summary, title="Validation-Set Summary (same best models)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import os
    import sys
    parser = argparse.ArgumentParser(description="Stock classification evaluation")
    parser.add_argument(
        "--use-validation-split",
        action="store_true",
        help="Run two expanding folds (validation + test). Default is a single held-out test fold.",
    )
    args = parser.parse_args()
    n_splits = 2 if args.use_validation_split else 1

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data import get_data_for_har
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
    from models.correlation_network import SquaredCorrelationNetwork, PartialCorrelationNetwork, MutualInformationNetwork
    from data.graph_cache import load_graph_data, graph_cache_exists

    SAMPLE_TICKERS = ["AAPL", "TSLA", "GOOG", "META", "MSFT", "NVDA", "NFLX", "AMZN"]
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    KNN_VALUES = [1, 2, 3, 4, 5]

    print("Loading and preprocessing data...")
    data_dict = get_data_for_har(100)
    tickers = list(data_dict.keys())
    graph_n_jobs = max(1, min(8, (os.cpu_count() or 1) - 1))
    cv_n_jobs = max(1, min(24, (os.cpu_count() or 1) - 1))
    print(f"Using {graph_n_jobs} worker processes for graph builds, {cv_n_jobs} for CV.")
    print(f"Using {n_splits} expanding-window fold(s) for evaluation.")

    # ── Offline graph builds — reuse cache from cross_val.py if available ────
    CACHE_DIR = RESULTS_DIR / "graph_cache"
    _tags = ["sqcorr", "pcorr", "exp", "mi"]
    _all_cached = all(graph_cache_exists(CACHE_DIR, t, KNN_VALUES) for t in _tags)
    cache_loaded = False

    if _all_cached:
        print("\nLoading cached graph features from regression run...")
        data_dicts_net, nets_sq = load_graph_data(CACHE_DIR, "sqcorr", KNN_VALUES)
        data_dicts_pcorr, nets_pc = load_graph_data(CACHE_DIR, "pcorr", KNN_VALUES)
        data_dicts_exp, nets_exp = load_graph_data(CACHE_DIR, "exp", KNN_VALUES)
        data_dicts_mi, nets_mi = load_graph_data(CACHE_DIR, "mi", KNN_VALUES)
        cache_loaded = True
        print("  Graph cache loaded — skipping expensive fit_transform.")
    else:
        print("\nGraph cache not found — building from scratch...")
        print("  (Run cross_val.py first to populate the cache.)")
        data_dicts_net: dict = {}
        nets_sq: dict = {}
        for k_val in KNN_VALUES:
            net_k = SquaredCorrelationNetwork(
                window=60, step=1, save_step=5, n_jobs=graph_n_jobs, graph_type="knn", k=k_val,
                feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            )
            data_dicts_net[k_val] = net_k.fit_transform(data_dict)
            nets_sq[k_val] = net_k
            print(f"  [SqCorr] k={k_val}: {net_k.n_all_snapshots_} total graphs, {net_k.n_snapshots_} saved.")

        data_dicts_pcorr: dict = {}
        nets_pc: dict = {}
        for k_val in KNN_VALUES:
            net_pk = PartialCorrelationNetwork(
                window=60, step=1, save_step=5, n_jobs=graph_n_jobs, graph_type="knn", k=k_val, shrinkage=0.1,
                feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            )
            data_dicts_pcorr[k_val] = net_pk.fit_transform(data_dict)
            nets_pc[k_val] = net_pk
            print(f"  [PCorr]  k={k_val}: {net_pk.n_all_snapshots_} total graphs, {net_pk.n_snapshots_} saved.")

        data_dicts_exp: dict = {}
        nets_exp: dict = {}
        for k_val in KNN_VALUES:
            net_exp = SquaredCorrelationNetwork(
                window=60, step=1, save_step=5, n_jobs=graph_n_jobs, graph_type="knn", k=k_val,
                feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
                idw_kernel="exp", exp_lambda=5.0,
            )
            data_dicts_exp[k_val] = net_exp.fit_transform(data_dict)
            nets_exp[k_val] = net_exp
            print(f"  [ExpKernel] k={k_val}: {net_exp.n_all_snapshots_} total graphs, {net_exp.n_snapshots_} saved.")

        data_dicts_mi: dict = {}
        nets_mi: dict = {}
        for k_val in KNN_VALUES:
            net_mi = MutualInformationNetwork(
                window=60, step=1, save_step=5, n_jobs=graph_n_jobs, graph_type="knn", k=k_val,
                n_bins=10,
                feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            )
            data_dicts_mi[k_val] = net_mi.fit_transform(data_dict)
            nets_mi[k_val] = net_mi
            print(f"  [MI]     k={k_val}: {net_mi.n_all_snapshots_} total graphs, {net_mi.n_snapshots_} saved.")

        from data.graph_cache import save_graph_data

        print("\nSaving graph cache for future classification runs...")
        save_graph_data(data_dicts_net, nets_sq, CACHE_DIR, "sqcorr")
        save_graph_data(data_dicts_pcorr, nets_pc, CACHE_DIR, "pcorr")
        save_graph_data(data_dicts_exp, nets_exp, CACHE_DIR, "exp")
        save_graph_data(data_dicts_mi, nets_mi, CACHE_DIR, "mi")
        print("  Graph cache saved to", CACHE_DIR)

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
        "DCC-GARCH-Logit": {
            "DCC-GARCH-Logit (C=0.1)":       (DCCGARCHSpikeClassifier(C=0.1), False),
            "DCC-GARCH-Logit (C=1.0)":       (DCCGARCHSpikeClassifier(C=1.0), False),
        },
    }

    def _network_clf_models() -> Dict[str, Any]:
        """Fresh network classifier instances (needed per k-value run)."""
        return _with_no_outlier_variants({
            "NetHAR-Logit (C=0.1)":           (NetworkHARClassifier(C=0.1),             False),
            "NetHAR-Logit (C=1.0)":           (NetworkHARClassifier(C=1.0),             False),
            "NetHAR-Logit (C=10.0)":          (NetworkHARClassifier(C=10.0),            False),
            "NetHAR-Logit (C=100.0)":         (NetworkHARClassifier(C=100.0),           False),
            "NetVAR-Logit (C=1,a=0.1,b=1)":   (NetworkVARClassifier(C_stage1=1.0,
                                                                     stage2_alpha=0.1,
                                                                     correction_bound=1.0), False),
            "NetVAR-Logit (C=1,a=0.5,b=2)":   (NetworkVARClassifier(C_stage1=1.0,
                                                                     stage2_alpha=0.5,
                                                                     correction_bound=2.0), False),
            "NetVAR-Logit (C=1,a=1.0,b=2)":   (NetworkVARClassifier(C_stage1=1.0,
                                                                     stage2_alpha=1.0,
                                                                     correction_bound=2.0), False),
            "NetVAR-Logit (C=1,a=0.01,b=1)":  (NetworkVARClassifier(C_stage1=1.0,
                                                                     stage2_alpha=0.01,
                                                                     correction_bound=1.0), False),
            "NetVAR-Logit (C=10,a=0.1,b=1)":  (NetworkVARClassifier(C_stage1=10.0,
                                                                     stage2_alpha=0.1,
                                                                     correction_bound=1.0), False),
            "NetVAR-Logit (C=10,a=0.5,b=2)":  (NetworkVARClassifier(C_stage1=10.0,
                                                                     stage2_alpha=0.5,
                                                                     correction_bound=2.0), False),
            "NetVAR-Logit (C=10,a=1.0,b=2)":  (NetworkVARClassifier(C_stage1=10.0,
                                                                     stage2_alpha=1.0,
                                                                     correction_bound=2.0), False),
        })

    def _network_clf_models_clustering() -> Dict[str, Any]:
        """Network classifier models with clustering features enabled."""
        return _with_no_outlier_variants({
            "NetHAR+C-Logit (C=0.1)":          (NetworkHARClassifier(C=0.1,   use_clustering=True), False),
            "NetHAR+C-Logit (C=1.0)":          (NetworkHARClassifier(C=1.0,   use_clustering=True), False),
            "NetHAR+C-Logit (C=10.0)":         (NetworkHARClassifier(C=10.0,  use_clustering=True), False),
            "NetHAR+C-Logit (C=100.0)":        (NetworkHARClassifier(C=100.0, use_clustering=True), False),
            "NetVAR+C-Logit (C=1,a=0.1,b=1)":  (NetworkVARClassifier(C_stage1=1.0,
                                                                      stage2_alpha=0.1,
                                                                      correction_bound=1.0,
                                                                      use_clustering=True), False),
            "NetVAR+C-Logit (C=1,a=0.5,b=2)":  (NetworkVARClassifier(C_stage1=1.0,
                                                                      stage2_alpha=0.5,
                                                                      correction_bound=2.0,
                                                                      use_clustering=True), False),
            "NetVAR+C-Logit (C=10,a=0.1,b=1)": (NetworkVARClassifier(C_stage1=10.0,
                                                                      stage2_alpha=0.1,
                                                                      correction_bound=1.0,
                                                                      use_clustering=True), False),
            "NetVAR+C-Logit (C=10,a=0.5,b=2)": (NetworkVARClassifier(C_stage1=10.0,
                                                                      stage2_alpha=0.5,
                                                                      correction_bound=2.0,
                                                                      use_clustering=True), False),
        })

    def _network_clf_models_sign_split() -> Dict[str, Any]:
        """Network classifier models with sign-split IDW features."""
        return _with_no_outlier_variants({
            "NetHAR-Split-Logit (C=0.1)":          (NetworkHARClassifier(C=0.1,   use_sign_split=True), False),
            "NetHAR-Split-Logit (C=1.0)":          (NetworkHARClassifier(C=1.0,   use_sign_split=True), False),
            "NetHAR-Split-Logit (C=10.0)":         (NetworkHARClassifier(C=10.0,  use_sign_split=True), False),
            "NetHAR-Split-Logit (C=100.0)":        (NetworkHARClassifier(C=100.0, use_sign_split=True), False),
            "NetVAR-Split-Logit (C=1,a=0.1,b=1)":  (NetworkVARClassifier(
                C_stage1=1.0,
                stage2_alpha=0.1,
                correction_bound=1.0,
                use_sign_split=True,
            ), False),
            "NetVAR-Split-Logit (C=1,a=0.5,b=2)":  (NetworkVARClassifier(
                C_stage1=1.0,
                stage2_alpha=0.5,
                correction_bound=2.0,
                use_sign_split=True,
            ), False),
            "NetVAR-Split-Logit (C=10,a=0.1,b=1)": (NetworkVARClassifier(
                C_stage1=10.0,
                stage2_alpha=0.1,
                correction_bound=1.0,
                use_sign_split=True,
            ), False),
        })

    def _network_clf_models_sign_split_clustering() -> Dict[str, Any]:
        """Sign-split network classifier models with clustering features."""
        return _with_no_outlier_variants({
            "NetHAR+CSplit-Logit (C=0.1)":          (NetworkHARClassifier(
                C=0.1,
                use_clustering=True,
                use_sign_split=True,
            ), False),
            "NetHAR+CSplit-Logit (C=1.0)":          (NetworkHARClassifier(
                C=1.0,
                use_clustering=True,
                use_sign_split=True,
            ), False),
            "NetHAR+CSplit-Logit (C=10.0)":         (NetworkHARClassifier(
                C=10.0,
                use_clustering=True,
                use_sign_split=True,
            ), False),
            "NetVAR+CSplit-Logit (C=1,a=0.1,b=1)":  (NetworkVARClassifier(
                C_stage1=1.0,
                stage2_alpha=0.1,
                correction_bound=1.0,
                use_clustering=True,
                use_sign_split=True,
            ), False),
            "NetVAR+CSplit-Logit (C=1,a=0.5,b=2)":  (NetworkVARClassifier(
                C_stage1=1.0,
                stage2_alpha=0.5,
                correction_bound=2.0,
                use_clustering=True,
                use_sign_split=True,
            ), False),
            "NetVAR+CSplit-Logit (C=10,a=0.1,b=1)": (NetworkVARClassifier(
                C_stage1=10.0,
                stage2_alpha=0.1,
                correction_bound=1.0,
                use_clustering=True,
                use_sign_split=True,
            ), False),
        })

    def _learned_weight_clf_models(k_val: int) -> Dict[str, Any]:
        """Learned m×k weight-matrix classifiers for a given k."""
        models: Dict[str, Any] = {}
        for m_val in range(1, k_val):
            models[f"LearnedW-Logit (m={m_val}, C=0.1)"] = (
                LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=0.1), False)
            models[f"LearnedW-Logit (m={m_val}, C=1.0)"] = (
                LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=1.0), False)
            models[f"LearnedW-Logit (m={m_val}, C=10.0)"] = (
                LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=10.0), False)
            models[f"LearnedW-Logit (m={m_val}, C=100.0)"] = (
                LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=100.0), False)
        return _with_no_outlier_variants(models)

    def _learned_weight_clf_clustering_models(k_val: int) -> Dict[str, Any]:
        """Learned m×k weight-matrix classifiers with clustering features."""
        models: Dict[str, Any] = {}
        for m_val in range(1, k_val):
            models[f"LearnedW+C-Logit (m={m_val}, C=0.1)"] = (
                LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=0.1,  use_clustering=True), False)
            models[f"LearnedW+C-Logit (m={m_val}, C=1.0)"] = (
                LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=1.0,  use_clustering=True), False)
            models[f"LearnedW+C-Logit (m={m_val}, C=10.0)"] = (
                LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=10.0, use_clustering=True), False)
            models[f"LearnedW+C-Logit (m={m_val}, C=100.0)"] = (
                LearnedWeightNetworkHARClassifier(k=k_val, m=m_val, C=100.0, use_clustering=True), False)
        return _with_no_outlier_variants(models)

    # ── Run HAR-Logit baselines ──────────────────────────────────────────────
    print(f"\nRunning baseline classifiers on {len(tickers)} tickers...")
    metrics_df, pred_store, all_params = classification_cv_multi(
        data_dict,
        baseline_catalogue,
        tickers,
        n_splits=n_splits,
        sample_tickers=SAMPLE_TICKERS,
        spike_quantile=0.8,
        save_params=True,
        num_workers=cv_n_jobs,
    )
    all_extra_metrics: list = []

    # Save graph snapshots for interpretability
    if cache_loaded:
        print("\nSkipping classification graph snapshot export; cached regression artifacts already exist.")
    else:
        print("\nSaving graph snapshots...")
        GRAPHS_DIR = RESULTS_DIR / "graphs"
        FEATURES_DIR = RESULTS_DIR / "feature_snapshots"
        for k_val in KNN_VALUES:
            save_graph_snapshots(nets_sq[k_val], GRAPHS_DIR, f"clf_sqcorr_k{k_val}")
            save_graph_snapshots(nets_pc[k_val], GRAPHS_DIR, f"clf_pcorr_k{k_val}")
            save_graph_snapshots(nets_exp[k_val], GRAPHS_DIR, f"clf_expkernel_k{k_val}")
            save_graph_snapshots(nets_mi[k_val], GRAPHS_DIR, f"clf_mi_k{k_val}")

            save_feature_snapshots(
                data_dicts_net[k_val], FEATURES_DIR, f"clf_sqcorr_k{k_val}",
                tickers=SAMPLE_TICKERS,
            )
            save_feature_snapshots(
                data_dicts_pcorr[k_val], FEATURES_DIR, f"clf_pcorr_k{k_val}",
                tickers=SAMPLE_TICKERS,
            )
            save_feature_snapshots(
                data_dicts_exp[k_val], FEATURES_DIR, f"clf_expkernel_k{k_val}",
                tickers=SAMPLE_TICKERS,
            )
            save_feature_snapshots(
                data_dicts_mi[k_val], FEATURES_DIR, f"clf_mi_k{k_val}",
                tickers=SAMPLE_TICKERS,
            )

    # ── Squared-correlation network classifiers ───────────────────────────────
    print("\nRunning network classifiers (SqCorr, k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        net_cat: Dict[str, Dict[str, Any]] = {
            f"Network [k={k_val}]": _network_clf_models()
        }
        dd_net = data_dicts_net[k_val]
        m_k, ps_k, params_k = classification_cv_multi(
            dd_net, net_cat, list(dd_net.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
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
        m_pk, ps_pk, params_pk = classification_cv_multi(
            dd_pc, pc_cat, list(dd_pc.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
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
        m_ek, ps_ek, params_ek = classification_cv_multi(
            dd_exp, exp_cat, list(dd_exp.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
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
        m_cl, ps_cl, params_cl = classification_cv_multi(
            dd_sq, cl_cat, list(dd_sq.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
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
        m_ec, ps_ec, params_ec = classification_cv_multi(
            dd_exp, ec_cat, list(dd_exp.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
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
        m_mi, ps_mi, params_mi = classification_cv_multi(
            dd_mi, mi_cat, list(dd_mi.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_mi)
        all_params.extend(params_mi)
        for t in ps_mi:
            new_cols = [c for c in ps_mi[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_mi[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_mi[t]

    # ── Sign-split benchmark classifiers ────────────────────────────────────
    print("\nRunning sign-split feature classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        split_cat: Dict[str, Dict[str, Any]] = {
            f"SplitFeatures [k={k_val}]": _network_clf_models_sign_split()
        }
        dd_sq = data_dicts_net[k_val]
        m_sp, ps_sp, params_sp = classification_cv_multi(
            dd_sq, split_cat, list(dd_sq.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_sp)
        all_params.extend(params_sp)
        for t in ps_sp:
            new_cols = [c for c in ps_sp[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_sp[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_sp[t]

    print("\nRunning sign-split + clustering classifiers (k=1..5)...")
    for k_val in KNN_VALUES:
        splitc_cat: Dict[str, Dict[str, Any]] = {
            f"Split+Clustering [k={k_val}]": _network_clf_models_sign_split_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        m_sc, ps_sc, params_sc = classification_cv_multi(
            dd_exp, splitc_cat, list(dd_exp.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_sc)
        all_params.extend(params_sc)
        for t in ps_sc:
            new_cols = [c for c in ps_sc[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_sc[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_sc[t]

    # ── Learned-weight classifiers (SqCorr data) ─────────────────────────────
    print("\nRunning learned-weight classifiers (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lw_cat: Dict[str, Dict[str, Any]] = {
            f"LearnedWeight [k={k_val}]": _learned_weight_clf_models(k_val)
        }
        dd_sq = data_dicts_net[k_val]
        m_lw, ps_lw, params_lw = classification_cv_multi(
            dd_sq, lw_cat, list(dd_sq.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lw)
        all_params.extend(params_lw)
        for t in ps_lw:
            new_cols = [c for c in ps_lw[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_lw[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_lw[t]

    # ── Learned-weight classifiers (PCorr data) ──────────────────────────────
    print("\nRunning learned-weight classifiers (PCorr, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lw_pc_cat: Dict[str, Dict[str, Any]] = {
            f"PCorr LearnedWeight [k={k_val}]": _learned_weight_clf_models(k_val)
        }
        dd_pc = data_dicts_pcorr[k_val]
        m_lwp, ps_lwp, params_lwp = classification_cv_multi(
            dd_pc, lw_pc_cat, list(dd_pc.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwp)
        all_params.extend(params_lwp)
        for t in ps_lwp:
            new_cols = [c for c in ps_lwp[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_lwp[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_lwp[t]

    # ── Learned-weight + clustering classifiers (SqCorr data) ────────────────
    print("\nRunning learned-weight + clustering classifiers (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lwc_cat: Dict[str, Dict[str, Any]] = {
            f"LW+Clustering [k={k_val}]": _learned_weight_clf_clustering_models(k_val)
        }
        dd_sq = data_dicts_net[k_val]
        m_lwc, ps_lwc, params_lwc = classification_cv_multi(
            dd_sq, lwc_cat, list(dd_sq.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwc)
        all_params.extend(params_lwc)
        for t in ps_lwc:
            new_cols = [c for c in ps_lwc[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_lwc[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_lwc[t]

    # ── Learned-weight + clustering classifiers (PCorr data) ─────────────────
    print("\nRunning learned-weight + clustering classifiers (PCorr, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lwc_pc_cat: Dict[str, Dict[str, Any]] = {
            f"PCorr LW+Clustering [k={k_val}]": _learned_weight_clf_clustering_models(k_val)
        }
        dd_pc = data_dicts_pcorr[k_val]
        m_lwcp, ps_lwcp, params_lwcp = classification_cv_multi(
            dd_pc, lwc_pc_cat, list(dd_pc.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwcp)
        all_params.extend(params_lwcp)
        for t in ps_lwcp:
            new_cols = [c for c in ps_lwcp[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_lwcp[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_lwcp[t]

    # ── PCorr sign-split classifiers ─────────────────────────────────────────
    print("\nRunning PCorr sign-split classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        ps_cat: Dict[str, Dict[str, Any]] = {
            f"PCorr Split [k={k_val}]": _network_clf_models_sign_split()
        }
        dd_pc = data_dicts_pcorr[k_val]
        m_ps, ps_ps, params_ps = classification_cv_multi(
            dd_pc, ps_cat, list(dd_pc.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_ps)
        all_params.extend(params_ps)
        for t in ps_ps:
            new_cols = [c for c in ps_ps[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_ps[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_ps[t]

    # ── ExpKernel sign-split classifiers ─────────────────────────────────────
    print("\nRunning ExpKernel sign-split classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        es_cat: Dict[str, Dict[str, Any]] = {
            f"ExpKernel Split [k={k_val}]": _network_clf_models_sign_split()
        }
        dd_exp = data_dicts_exp[k_val]
        m_es, ps_es, params_es = classification_cv_multi(
            dd_exp, es_cat, list(dd_exp.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_es)
        all_params.extend(params_es)
        for t in ps_es:
            new_cols = [c for c in ps_es[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_es[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_es[t]

    # ── MI sign-split classifiers ────────────────────────────────────────────
    print("\nRunning MI sign-split classifiers (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        ms_cat: Dict[str, Dict[str, Any]] = {
            f"MI Split [k={k_val}]": _network_clf_models_sign_split()
        }
        dd_mi = data_dicts_mi[k_val]
        m_ms, ps_ms, params_ms = classification_cv_multi(
            dd_mi, ms_cat, list(dd_mi.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_ms)
        all_params.extend(params_ms)
        for t in ps_ms:
            new_cols = [c for c in ps_ms[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_ms[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_ms[t]

    # ── MI learned-weight classifiers ────────────────────────────────────────
    print("\nRunning MI learned-weight classifiers (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lw_mi_cat: Dict[str, Dict[str, Any]] = {
            f"MI LearnedWeight [k={k_val}]": _learned_weight_clf_models(k_val)
        }
        dd_mi = data_dicts_mi[k_val]
        m_lwm, ps_lwm, params_lwm = classification_cv_multi(
            dd_mi, lw_mi_cat, list(dd_mi.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwm)
        all_params.extend(params_lwm)
        for t in ps_lwm:
            new_cols = [c for c in ps_lwm[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_lwm[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_lwm[t]

    # ── MI learned-weight + clustering classifiers ───────────────────────────
    print("\nRunning MI learned-weight + clustering classifiers (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lwc_mi_cat: Dict[str, Dict[str, Any]] = {
            f"MI LW+Clustering [k={k_val}]": _learned_weight_clf_clustering_models(k_val)
        }
        dd_mi = data_dicts_mi[k_val]
        m_lwcm, ps_lwcm, params_lwcm = classification_cv_multi(
            dd_mi, lwc_mi_cat, list(dd_mi.keys()),
            n_splits=n_splits, sample_tickers=SAMPLE_TICKERS, spike_quantile=0.8,
            save_params=True,
            num_workers=cv_n_jobs,
        )
        all_extra_metrics.append(m_lwcm)
        all_params.extend(params_lwcm)
        for t in ps_lwcm:
            new_cols = [c for c in ps_lwcm[t].columns if c != "Y_true_spike"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps_lwcm[t][new_cols], how="outer")
            else:
                pred_store[t] = ps_lwcm[t]

    # ── Merge all results ─────────────────────────────────────────────────────
    metrics_df = pd.concat([metrics_df] + all_extra_metrics, ignore_index=True)

    # Save full (un-coalesced) results for later reload
    summary_raw = summarize_classification(metrics_df)
    save_classification_results(metrics_df, summary_raw, RESULTS_DIR)
    save_model_params(all_params, RESULTS_DIR, "classification_model_params.json")
    save_classification_prediction_store(pred_store, RESULTS_DIR)

    # Coalesce [k=N] categories for cleaner printing
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
        summary_title = "Classification Summary (all stocks)"
    else:
        print("\n" + "=" * 80)
        print("TEST SUMMARY  (Fold 0 — held-out final evaluation)")
        print("=" * 80)
        test_metrics = metrics_coalesced[metrics_coalesced["Fold"] == 0]
        test_summary = summarize_classification(test_metrics)
        print_classification_summary(test_summary, title="Test Summary (all models)")
        print_best_classifier(test_summary)
        summary_title = "Classification Summary — single hold-out fold (all stocks)"

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
    Fast smoke-test entry point for the stock classification pipeline.

    Runs one model per baseline family + two network models (NetHAR, NetVAR)
    on a single SqCorr graph at k=3.  Intended to verify the full pipeline
    works without committing to a multi-hour full run.

    To restore the full experiment, replace the ``sanity()`` call in the
    ``__main__`` block below with ``main()``.
    """
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data import get_data_for_har
    from models.baselines_classification import (
        HARLogitClassifier,
        HARExtendedLogitClassifier,
        RegimeSwitchingHARLogitClassifier,
        DCCGARCHSpikeClassifier,
    )
    from models.network_models_classification import (
        NetworkHARClassifier,
        NetworkVARClassifier,
    )
    from models.correlation_network import SquaredCorrelationNetwork
    from data.graph_cache import load_graph_data, graph_cache_exists

    SAMPLE_TICKERS = ["AAPL", "TSLA", "GOOG", "META", "MSFT", "NVDA", "NFLX", "AMZN"]
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    K_VAL = 3

    print("Loading and preprocessing data...")
    data_dict = get_data_for_har(100)
    tickers = list(data_dict.keys())
    graph_n_jobs = max(1, min(8, (os.cpu_count() or 1) - 1))
    cv_n_jobs = max(1, min(24, (os.cpu_count() or 1) - 1))
    print(f"  {len(tickers)} tickers — sanity mode (1 model/family, SqCorr k={K_VAL} only)")

    # ── Graph: reuse cached SqCorr from regression run if available ──────────
    CACHE_DIR = RESULTS_DIR / "graph_cache"
    cache_loaded = False
    if graph_cache_exists(CACHE_DIR, "sqcorr", [K_VAL]):
        print("\nLoading cached SqCorr graph features (k={K_VAL})...")
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
        "HAR-Logit":             {"HAR-Logit (C=1.0)":          (HARLogitClassifier(C=1.0),                                  False)},
        "HAR-Ext-Logit":         {"HAR-Ext-Logit (C=1.0)":      (HARExtendedLogitClassifier(C=1.0),                          False)},
        "RegimeSwitching-Logit": {"RegHAR-Logit (p50)":         (RegimeSwitchingHARLogitClassifier(regime_percentile=0.5),    False)},
        "DCC-GARCH-Logit":       {"DCC-GARCH-Logit (C=1.0)":    (DCCGARCHSpikeClassifier(C=1.0),                             False)},
    }

    print(f"\nRunning baseline classifiers on {len(tickers)} tickers...")
    metrics_df, pred_store, all_params = classification_cv_multi(
        data_dict, baseline_catalogue, tickers,
        n_splits=1,
        sample_tickers=SAMPLE_TICKERS,
        spike_quantile=0.8,
        save_params=True,
        num_workers=cv_n_jobs,
    )

    # ── Single SqCorr network run at k=3 ─────────────────────────────────────
    net_catalogue: Dict[str, Dict[str, Any]] = {
        f"Network [k={K_VAL}]": {
            "NetHAR-Logit (C=1.0)":         (NetworkHARClassifier(C=1.0),                                False),
            "NetVAR-Logit (C=1,a=0.1,b=1)": (NetworkVARClassifier(C_stage1=1.0, stage2_alpha=0.1,
                                                                    correction_bound=1.0),                False),
        },
    }
    dd_net = data_dicts_net[K_VAL]
    print(f"\nRunning network classifiers (SqCorr, k={K_VAL})...")
    m_net, ps_net, params_net = classification_cv_multi(
        dd_net, net_catalogue, list(dd_net.keys()),
        n_splits=1,
        sample_tickers=SAMPLE_TICKERS,
        spike_quantile=0.8,
        save_params=True,
        num_workers=cv_n_jobs,
    )
    for t in ps_net:
        new_cols = [c for c in ps_net[t].columns if c != "Y_true_spike"]
        if t in pred_store:
            pred_store[t] = pred_store[t].join(ps_net[t][new_cols], how="outer")
        else:
            pred_store[t] = ps_net[t]
    all_params.extend(params_net)

    # ── Aggregate and save ────────────────────────────────────────────────────
    metrics_df = pd.concat([metrics_df, m_net], ignore_index=True)
    summary = summarize_classification(metrics_df)
    save_classification_results(metrics_df, summary, RESULTS_DIR)
    save_model_params(all_params, RESULTS_DIR, "classification_model_params.json")
    save_classification_prediction_store(pred_store, RESULTS_DIR)

    print_classification_summary(summary, title="Stock Classification (sanity — 1 model/family, SqCorr k=3)")
    print_best_classifier(summary)
    print_compact_clf_leaderboard(summary)


if __name__ == "__main__":
    # sanity()
    main()
