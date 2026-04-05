import copy
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.baselines import (
    HARLogRegressor, HARExtendedLogRegressor, ARIMALogY, GARCHWeeklyRV,
    DCCGARCHWeeklyRV,
    RegimeSwitchingHARLogRegressor,
)
from models.network_models import (
    NetworkHARRegressor,
    NetworkVARRegressor,
    LearnedWeightNetworkHARRegressor,
)
from data.preprocess import remove_outliers
from evaluation.interpretability import (
    extract_model_params,
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
    load_and_print_results,
)

# Module-level data store.  Populated in cross_val_multi() before workers are
# forked so that DataFrames are inherited copy-on-write — zero pickle cost.
_CV_DATA: dict = {}


def expanding_folds(
    index: "pd.Index",
    n_splits: int = 1,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
):
    """
    Expanding-window folds with contiguous, non-overlapping test blocks.

    Folds are generated from the *end* of the series backwards so that the
    most recent data is always evaluated:
      Fold 0: train = index[:n-test_size],  test = index[n-test_size:]
      Fold 1: train = index[:n-2*test_size], test = index[n-2*test_size:n-test_size]
      ...

    Yields (train_idx, test_idx) as index label arrays (not integer positions).

    Parameters
    ----------
    n_splits      : Number of folds.  Use 1 for a single hold-out evaluation.
    test_size     : Number of observations per test fold (default = 1 year).
    min_train_size: Minimum training observations; folds that would fall below
                    this threshold are silently skipped.
    """
    n = len(index)
    if n < min_train_size + test_size:
        raise ValueError(
            f"Not enough data (n={n}) for min_train_size={min_train_size} + test_size={test_size}."
        )

    for k in range(n_splits):
        test_end   = n - k * test_size
        test_start = test_end - test_size
        train_end  = test_start

        if train_end < min_train_size:
            break

        train_idx = index[:train_end]
        test_idx  = index[test_start:test_end]
        yield train_idx, test_idx


def eval_regression(y_true, y_pred, eps: float = 1e-8) -> dict:
    """
    Evaluate predictions in both original and log scale.

    Log-scale metrics (R2_log, RMSE_log, MAE_log) are often more meaningful
    for heavy-tailed realized-variance targets because outlier spikes do not
    dominate the score.
    """
    log_true = np.log(np.clip(y_true, eps, None))
    log_pred = np.log(np.clip(y_pred, eps, None))
    return {
        "R2":       float(r2_score(y_true, y_pred)),
        "RMSE":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "R2_log":   float(r2_score(log_true, log_pred)),
        "RMSE_log": float(np.sqrt(mean_squared_error(log_true, log_pred))),
        "MAE_log":  float(mean_absolute_error(log_true, log_pred)),
    }


def fit_predict_model(
    model,
    X_train: "pd.DataFrame",
    X_test: "pd.DataFrame",
) -> "np.ndarray":
    """
    Fit *model* on X_train and return predictions for X_test.

    ``model.features`` must list the column names needed by the model.
    ``X_train["log_Y"]`` is used as the training target (log of forward RV).
    Predictions are returned in original variance scale.
    """
    X_train_feat = X_train[model.features]
    X_test_feat  = X_test[model.features]
    y_train      = X_train["log_Y"]
    model.fit(X_train_feat, y_train)
    return model.predict(X_test_feat)


def run_benchmarks_multi_fold(
    data_dict: "dict[str, pd.DataFrame]",
    model_catalogue: "Dict[str, Dict[str, Any]]",
    tickers: "list[str]",
    n_splits: int = 1,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
    sample_tickers: "list[str] | None" = None,
    save_params: bool = False,
) -> "tuple[pd.DataFrame, dict, list] | tuple[pd.DataFrame, dict]":
    """
    Run expanding-window cross-validation for a categorised model catalogue.

    Parameters
    ----------
    data_dict       : {ticker: DataFrame} with HAR features, log_Y, Y_fwd, and
                      any columns required by the models.
    model_catalogue : Nested dict  {category: {model_name: model_entry}}
                      where model_entry is either a model instance or a
                      (model_instance, remove_outliers_bool) tuple.
                      Grouping models into categories allows the plotting code
                      to pick the best model per category automatically.
    n_splits        : Number of expanding-window folds.
    sample_tickers  : Store per-row predictions for these tickers for plotting.

    Returns
    -------
    metrics_df : Long-form DataFrame with columns
                 [Category, Ticker, Model, Fold, R2, RMSE, MAE,
                  R2_log, RMSE_log, MAE_log].
    pred_store : {ticker: DataFrame} where columns are model names and rows
                 are test-fold dates.
    """
    rows: list = []
    pred_store: dict = {}
    params_store: list = []
    sample_set = set(sample_tickers) if sample_tickers else set()

    for t in tqdm(tickers, desc="Tickers", unit="ticker"):
        if t not in data_dict:
            continue

        feat = data_dict[t]
        if len(feat) < min_train_size + test_size:
            continue

        y_true = feat["Y_fwd"]

        if t in sample_set:
            pred_df = pd.DataFrame(index=feat.index, data={"Y_true": y_true})

        for fold_id, (train_idx, test_idx) in enumerate(
            expanding_folds(feat.index, n_splits, test_size, min_train_size)
        ):
            X_train = feat.loc[train_idx]
            X_test  = feat.loc[test_idx]

            for category, models in model_catalogue.items():
                for model_name, model_entry in models.items():
                    if isinstance(model_entry, tuple):
                        model_template, do_remove_outliers = model_entry
                    else:
                        model_template, do_remove_outliers = model_entry, False

                    model = copy.deepcopy(model_template)
                    X_train_fit = remove_outliers(X_train) if do_remove_outliers else X_train
                    y_pred = fit_predict_model(model, X_train_fit, X_test)
                    m = eval_regression(y_true.loc[test_idx].values, y_pred)
                    rows.append({
                        "Category": category,
                        "Ticker":   t,
                        "Model":    model_name,
                        "Fold":     fold_id,
                        **m,
                    })

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
                        pred_df.loc[test_idx, col_key] = y_pred

        if t in sample_set:
            pred_store[t] = pred_df

    metrics_df = pd.DataFrame(rows)
    if save_params:
        return metrics_df, pred_store, params_store
    return metrics_df, pred_store


def _run_single_benchmark_task(task: "dict[str, Any]") -> "dict[str, Any]":
    """Execute one (ticker, fold, model) benchmark task."""
    feat = _CV_DATA[task["ticker"]]  # inherited via fork — no deserialisation cost
    train_idx = task["train_idx"]
    test_idx = task["test_idx"]

    X_train = feat.loc[train_idx]
    X_test = feat.loc[test_idx]

    model = copy.deepcopy(task["model_template"])
    X_train_fit = remove_outliers(X_train) if task["do_remove_outliers"] else X_train
    y_pred = fit_predict_model(model, X_train_fit, X_test)
    metrics = eval_regression(feat.loc[test_idx, "Y_fwd"].values, y_pred)

    result = {
        "row": {
            "Category": task["category"],
            "Ticker": task["ticker"],
            "Model": task["model_name"],
            "Fold": task["fold_id"],
            **metrics,
        },
        "prediction": None,
        "params": None,
    }

    if task["store_predictions"]:
        result["prediction"] = {
            "ticker": task["ticker"],
            "col_key": f"[{task['category']}] {task['model_name']}",
            "test_idx": test_idx,
            "y_pred": y_pred,
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


def cross_val_multi(
    data_dict: "dict[str, pd.DataFrame]",
    model_catalogue: "Dict[str, Dict[str, Any]]",
    tickers: "list[str]",
    n_splits: int = 1,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
    sample_tickers: "list[str] | None" = None,
    save_params: bool = False,
    num_threads: int = 24,
) -> "tuple[pd.DataFrame, dict, list] | tuple[pd.DataFrame, dict]":
    """
    Threaded expanding-window cross-validation with dynamic load balancing.

    Work is split at the (ticker, fold, model) level so slow models naturally
    occupy threads longer while faster tasks continue to drain from the queue.
    This is more balanced than assigning a fixed subset of models or tickers to
    each worker up front.
    """
    rows: list = []
    pred_store: dict = {}
    params_store: list = []
    sample_set = set(sample_tickers) if sample_tickers else set()
    tasks: list = []

    for ticker in tickers:
        if ticker not in data_dict:
            continue

        feat = data_dict[ticker]
        if len(feat) < min_train_size + test_size:
            continue

        if ticker in sample_set:
            pred_store[ticker] = pd.DataFrame(index=feat.index, data={"Y_true": feat["Y_fwd"]})

        for fold_id, (train_idx, test_idx) in enumerate(
            expanding_folds(feat.index, n_splits, test_size, min_train_size)
        ):
            for category, models in model_catalogue.items():
                for model_name, model_entry in models.items():
                    if isinstance(model_entry, tuple):
                        model_template, do_remove_outliers = model_entry
                    else:
                        model_template, do_remove_outliers = model_entry, False

                    tasks.append(
                        {
                            "ticker": ticker,
                            # "feat" is intentionally omitted — workers look it
                            # up from _CV_DATA (inherited via fork) so the
                            # DataFrame is never serialised into individual tasks.
                            "fold_id": fold_id,
                            "train_idx": train_idx,
                            "test_idx": test_idx,
                            "category": category,
                            "model_name": model_name,
                            "model_template": model_template,
                            "do_remove_outliers": do_remove_outliers,
                            "store_predictions": ticker in sample_set,
                            "save_params": save_params,
                        }
                    )

    if not tasks:
        metrics_df = pd.DataFrame(rows)
        if save_params:
            return metrics_df, pred_store, params_store
        return metrics_df, pred_store

    # Use fork-based multiprocessing to bypass the GIL.
    # With "fork", child processes inherit _CV_DATA (set just above) via
    # copy-on-write — the DataFrames are never pickled into the task queue.
    global _CV_DATA
    _CV_DATA = data_dict

    max_workers = max(1, int(num_threads))
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_run_single_benchmark_task, tasks),
            total=len(tasks),
            desc="CV tasks",
            unit="task",
        ):
            rows.append(result["row"])

            pred_result = result["prediction"]
            if pred_result is not None:
                pred_store[pred_result["ticker"]].loc[pred_result["test_idx"], pred_result["col_key"]] = pred_result["y_pred"]

            if result["params"] is not None:
                params_store.append(result["params"])

    metrics_df = pd.DataFrame(rows)
    if save_params:
        return metrics_df, pred_store, params_store
    return metrics_df, pred_store


def save_results(
    metrics_df: pd.DataFrame,
    summary: pd.DataFrame,
    results_dir: Path,
) -> None:
    """
    Persist cross-validation outputs to disk.

    Files written
    -------------
    results_bench.json : fold-level metrics, one record per (Category, Ticker, Model, Fold).
    summary_bench.json : model-level summary aggregated across tickers, sorted by
                         median_R2_log descending.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / "results_bench.json"
    metrics_df.to_json(raw_path, orient="records", indent=2)
    print(f"Fold-level metrics  -> {raw_path}")

    summary_path = results_dir / "summary_bench.json"
    summary.reset_index().to_json(summary_path, orient="records", indent=2)
    print(f"Summary metrics     -> {summary_path}")


def save_prediction_store(
    pred_store: Dict[str, pd.DataFrame],
    results_dir: Path,
    folder_name: str = "predictions_regression",
) -> None:
    """Persist per-ticker model forecasts for later plotting/analysis."""
    out_dir = Path(results_dir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for ticker, df in pred_store.items():
        if df is None or df.empty:
            continue
        out_path = out_dir / f"{ticker}_predictions.csv"
        df_out = df.copy()
        df_out.index = pd.to_datetime(df_out.index)
        df_out.index.name = "Date"
        df_out.to_csv(out_path)
        n_written += 1

    print(f"Prediction files    -> {out_dir}  ({n_written} tickers)")


def select_best_on_validation(
    metrics_df: pd.DataFrame,
    val_fold: int = 1,
    metric: str = "R2_log",
) -> pd.DataFrame:
    """
    Select the best model per category based on validation-fold performance.

    Returns a DataFrame with columns [Category, Model, median_{metric}].
    """
    val_rows = metrics_df[metrics_df["Fold"] == val_fold]
    if val_rows.empty:
        raise ValueError(f"No rows found for Fold={val_fold}")

    agg = (
        val_rows.groupby(["Category", "Model"])[metric]
        .median()
        .reset_index()
        .rename(columns={metric: f"median_{metric}"})
    )

    best_idx = agg.groupby("Category")[f"median_{metric}"].idxmax()
    best = agg.loc[best_idx, ["Category", "Model", f"median_{metric}"]].reset_index(drop=True)
    return best


def print_best_test_summary(
    metrics_df: pd.DataFrame,
    best_models: pd.DataFrame,
    test_fold: int = 0,
) -> None:
    """
    Print summary for only the best models (selected on validation)
    evaluated on the test fold.
    """
    test_rows = metrics_df[metrics_df["Fold"] == test_fold]
    best_set = set(zip(best_models["Category"], best_models["Model"]))
    mask = test_rows.apply(lambda r: (r["Category"], r["Model"]) in best_set, axis=1)
    test_best = test_rows[mask].copy()

    if test_best.empty:
        print("No test-fold results for the selected best models.")
        return

    summary = summarize_benchmarks(test_best)
    print("\n" + "=" * 80)
    print("BEST MODELS  (selected on validation fold, evaluated on test fold)")
    print("=" * 80)
    print_summary(summary, title="Test-Set Summary (best per category)")
    print_best_per_category(summary)
    print_compact_leaderboard(summary)

    # Also print val-fold metrics of the best models for comparison
    val_rows = metrics_df[metrics_df["Fold"] == 1]
    val_mask = val_rows.apply(lambda r: (r["Category"], r["Model"]) in best_set, axis=1)
    val_best = val_rows[val_mask].copy()
    if not val_best.empty:
        val_summary = summarize_benchmarks(val_best)
        print_summary(val_summary, title="Validation-Set Summary (same best models)")


def _with_no_outlier_variants(models: Dict[str, Any]) -> Dict[str, Any]:
    """Duplicate a model dictionary with training-only no-outlier variants."""
    augmented: Dict[str, Any] = {}
    for model_name, model_entry in models.items():
        if isinstance(model_entry, tuple):
            model_template, do_remove_outliers = model_entry
        else:
            model_template, do_remove_outliers = model_entry, False

        augmented[model_name] = (model_template, do_remove_outliers)
        if not do_remove_outliers:
            augmented[f"{model_name} (no outliers)"] = (model_template, True)
    return augmented


def main():
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data import get_data_for_har

    SAMPLE_TICKERS = ["AAPL", "TSLA", "GOOG", "META", "MSFT", "NVDA", "NFLX", "AMZN"]
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading and preprocessing data...")
    data_dict = get_data_for_har(100)
    tickers = list(data_dict.keys())
    graph_n_jobs = max(1, min(24, (os.cpu_count() or 1) - 1))
    print(f"Using {graph_n_jobs} worker processes for graph builds.")

    # ── Offline graph build ──────────────────────────────────────────────────
    # One SquaredCorrelationNetwork per k value, built ONCE on the full dataset.
    # Each snapshot at date t only uses returns r_{t-window+1}...r_t (past only),
    # so fitting on the full history does NOT introduce look-ahead leakage.
    print("\nBuilding correlation-network features offline (k=1..5)...")
    from models.correlation_network import SquaredCorrelationNetwork, PartialCorrelationNetwork, MutualInformationNetwork
    KNN_VALUES = [1, 2, 3, 4, 5]
    nets: dict = {}
    data_dicts_net: dict = {}
    for k_val in KNN_VALUES:
        net_k = SquaredCorrelationNetwork(
            window=60,
            step=1,
            save_step=5,
            n_jobs=graph_n_jobs,
            graph_type="knn",
            k=k_val,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_net[k_val] = net_k.fit_transform(data_dict)
        nets[k_val] = net_k
        print(f"  [SqCorr] k={k_val}: {net_k.n_all_snapshots_} total graphs, {net_k.n_snapshots_} saved.")
    # Save graph snapshots for interpretability
    print("\nSaving graph snapshots...")
    GRAPHS_DIR = RESULTS_DIR / "graphs"
    FEATURES_DIR = RESULTS_DIR / "feature_snapshots"
    for k_val in KNN_VALUES:
        save_graph_snapshots(nets[k_val], GRAPHS_DIR, f"sqcorr_k{k_val}")
        save_feature_snapshots(
            data_dicts_net[k_val], FEATURES_DIR, f"sqcorr_k{k_val}",
            tickers=SAMPLE_TICKERS,
        )

    print("\nBuilding partial-correlation network features offline (k=1..5)...")
    nets_pcorr: dict = {}
    data_dicts_pcorr: dict = {}
    for k_val in KNN_VALUES:
        net_pk = PartialCorrelationNetwork(
            window=60,
            step=1,
            save_step=5,
            n_jobs=graph_n_jobs,
            graph_type="knn",
            k=k_val,
            shrinkage=0.1,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_pcorr[k_val] = net_pk.fit_transform(data_dict)
        nets_pcorr[k_val] = net_pk
        print(f"  [PCorr]  k={k_val}: {net_pk.n_all_snapshots_} total graphs, {net_pk.n_snapshots_} saved.")

    for k_val in KNN_VALUES:
        save_graph_snapshots(nets_pcorr[k_val], GRAPHS_DIR, f"pcorr_k{k_val}")
        save_feature_snapshots(
            data_dicts_pcorr[k_val], FEATURES_DIR, f"pcorr_k{k_val}",
            tickers=SAMPLE_TICKERS,
        )

    # Exp-kernel: same SquaredCorrelationNetwork but with exp(-lambda*d) IDW
    print("\nBuilding exp-kernel correlation-network features offline (k=1..5)...")
    data_dicts_exp: dict = {}
    for k_val in KNN_VALUES:
        net_exp = SquaredCorrelationNetwork(
            window=60,
            step=1,
            save_step=5,
            n_jobs=graph_n_jobs,
            graph_type="knn",
            k=k_val,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
            idw_kernel="exp",
            exp_lambda=5.0,
        )
        data_dicts_exp[k_val] = net_exp.fit_transform(data_dict)
        print(f"  [ExpKernel] k={k_val}: {net_exp.n_all_snapshots_} total graphs, {net_exp.n_snapshots_} saved.")
        save_feature_snapshots(
            data_dicts_exp[k_val], FEATURES_DIR, f"expkernel_k{k_val}",
            tickers=SAMPLE_TICKERS,
        )

    # ── Mutual-information network build ─────────────────────────────────────
    print("\nBuilding mutual-information networks (k=1..5)...")
    nets_mi: dict = {}
    data_dicts_mi: dict = {}
    for k_val in KNN_VALUES:
        net_mi = MutualInformationNetwork(
            window=60,
            step=1,
            save_step=5,
            n_jobs=graph_n_jobs,
            graph_type="knn",
            k=k_val,
            n_bins=10,
            feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        )
        data_dicts_mi[k_val] = net_mi.fit_transform(data_dict)
        nets_mi[k_val] = net_mi
        print(f"  [MI]     k={k_val}: {net_mi.n_all_snapshots_} total graphs, {net_mi.n_snapshots_} saved.")

    for k_val in KNN_VALUES:
        save_graph_snapshots(nets_mi[k_val], GRAPHS_DIR, f"mi_k{k_val}")
        save_feature_snapshots(
            data_dicts_mi[k_val], FEATURES_DIR, f"mi_k{k_val}",
            tickers=SAMPLE_TICKERS,
        )

    # ── Model catalogue (category -> model dict) ─────────────────────────────
    # Each value is {model_display_name: (model_instance, strip_outliers_bool)}.
    # Models within the same category are compared against each other; the best
    # one per category is selected for the predictions plot.
    baseline_catalogue: Dict[str, Dict[str, Any]] = {
        "HAR": {
            "HAR":                        (HARLogRegressor(),                               False),
            "HAR (no outliers)":          (HARLogRegressor(),                               True),
            "HAR-Extended":               (HARExtendedLogRegressor(),                       False),
            "HAR-Extended (no outliers)": (HARExtendedLogRegressor(),                       True),
            "HAR-Lasso (a=0.01)":         (HARLogRegressor(lasso_alpha=0.01),               False),
            "HAR-Lasso (a=0.1)":          (HARLogRegressor(lasso_alpha=0.1),                False),
            "HAR-Ext-Lasso (a=0.01)":     (HARExtendedLogRegressor(lasso_alpha=0.01),       False),
            "HAR-Ext-Lasso (a=0.1)":      (HARExtendedLogRegressor(lasso_alpha=0.1),        False),
        },
        "ARIMA": {
            # ARMA baselines (d=0) -- exploit autocorrelation in log_Y levels
            "ARIMA(1,0,1)":               (ARIMALogY(order=(1, 0, 1)),                      False),
            "ARIMA(2,0,1)":               (ARIMALogY(order=(2, 0, 1)),                      False),
            "ARIMA(1,0,2)":               (ARIMALogY(order=(1, 0, 2)),                      False),
            "ARIMA(2,0,2)":               (ARIMALogY(order=(2, 0, 2)),                      False),
            # Integrated baselines (d=1) -- model first differences of log_Y
            "ARIMA(1,1,0)":               (ARIMALogY(order=(1, 1, 0)),                      False),
            "ARIMA(0,1,1)":               (ARIMALogY(order=(0, 1, 1)),                      False),
            "ARIMA(1,1,1)":               (ARIMALogY(order=(1, 1, 1)),                      False),
            "ARIMA(2,1,1)":               (ARIMALogY(order=(2, 1, 1)),                      False),
            "ARIMA(1,1,2)":               (ARIMALogY(order=(1, 1, 2)),                      False),
        },
        "GARCH": {
            "GARCH(1,1)":                 (GARCHWeeklyRV(p=1, q=1, horizon=5),              False),
            "GARCH(2,1)":                 (GARCHWeeklyRV(p=2, q=1, horizon=5),              False),
            "GARCH(1,2)":                 (GARCHWeeklyRV(p=1, q=2, horizon=5),              False),
            "GARCH(2,2)":                 (GARCHWeeklyRV(p=2, q=2, horizon=5),              False),
            "GARCH(3,2)":                 (GARCHWeeklyRV(p=3, q=2, horizon=5),              False),
            "GARCH(2,3)":                 (GARCHWeeklyRV(p=2, q=3, horizon=5),              False),
            "GARCH(3,3)":                 (GARCHWeeklyRV(p=3, q=3, horizon=5),              False),
            "DCC-GARCH(1,1)":             (DCCGARCHWeeklyRV(p=1, q=1, horizon=5),           False),
        },
        "RegimeSwitching": {
            "RegHAR (p50)": (RegimeSwitchingHARLogRegressor(regime_percentile=0.50), False),
            "RegHAR (p75)": (RegimeSwitchingHARLogRegressor(regime_percentile=0.75), False),
            "RegHAR (p90)": (RegimeSwitchingHARLogRegressor(regime_percentile=0.90), False),
            "RegHAR-Lasso (p50)": (RegimeSwitchingHARLogRegressor(lasso_alpha=0.01, regime_percentile=0.50), False),
            "RegHAR-Lasso (p75)": (RegimeSwitchingHARLogRegressor(lasso_alpha=0.01, regime_percentile=0.75), False),
        },
    }

    def _network_models() -> Dict[str, Any]:
        """Return fresh network model instances (needed per k-value run)."""
        return _with_no_outlier_variants({
            "NetHAR (Lasso a=0.20)":          (NetworkHARRegressor(lasso_alpha=0.20),                                          False),
            "NetHAR (Lasso a=0.10)":          (NetworkHARRegressor(lasso_alpha=0.10),                                          False),
            "NetHAR (Lasso a=0.05)":          (NetworkHARRegressor(lasso_alpha=0.05),                                          False),
            "NetHAR (Lasso a=0.01)":          (NetworkHARRegressor(lasso_alpha=0.01),                                          False),
            "NetHAR (OLS)":                   (NetworkHARRegressor(lasso_alpha=0.0),                                           False),
            "NetHAR (Ridge a=0.01)":          (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=0.01),                         False),
            "NetHAR (Ridge a=0.10)":          (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=0.10),                         False),
            "NetHAR (Ridge a=1.0)":           (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=1.0),                          False),
            "NetworkVAR (a=0.0, b=0.5)":      (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=0.5),                   False),
            "NetworkVAR (a=0.0, b=1.0)":      (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=1.0),                   False),
            "NetworkVAR (a=0.0, b=None)":     (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=None),                  False),
            "NetworkVAR (a=0.1, b=0.5)":      (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=0.5),                   False),
            "NetworkVAR (a=0.5, b=0.5)":      (NetworkVARRegressor(stage2_alpha=0.5,  correction_bound=0.5),                   False),
            "NetworkVAR (a=0.1, b=1.0)":      (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=1.0),                   False),
            "NetworkVAR (a=0.1, b=None)":     (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=None),                  False),
        })

    def _network_models_clustering() -> Dict[str, Any]:
        """Network models with clustering features enabled."""
        return _with_no_outlier_variants({
            "NetHAR+C (Lasso a=0.20)":        (NetworkHARRegressor(lasso_alpha=0.20, use_clustering=True),                    False),
            "NetHAR+C (Lasso a=0.10)":        (NetworkHARRegressor(lasso_alpha=0.10, use_clustering=True),                    False),
            "NetHAR+C (Lasso a=0.05)":        (NetworkHARRegressor(lasso_alpha=0.05, use_clustering=True),                    False),
            "NetHAR+C (Lasso a=0.01)":        (NetworkHARRegressor(lasso_alpha=0.01, use_clustering=True),                    False),
            "NetHAR+C (OLS)":                 (NetworkHARRegressor(lasso_alpha=0.0,  use_clustering=True),                    False),
            "NetHAR+C (Ridge a=0.01)":        (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.01,
                                                                    use_clustering=True),                                      False),
            "NetHAR+C (Ridge a=0.10)":        (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.10,
                                                                    use_clustering=True),                                      False),
            "NetHAR+C (Ridge a=1.0)":         (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=1.0,
                                                                    use_clustering=True),                                      False),
            "NetworkVAR+C (a=0.0, b=0.5)":    (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=0.5,
                                                                    use_clustering=True),                                      False),
            "NetworkVAR+C (a=0.0, b=1.0)":    (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=1.0,
                                                                    use_clustering=True),                                      False),
            "NetworkVAR+C (a=0.0, b=None)":   (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=None,
                                                                    use_clustering=True),                                      False),
            "NetworkVAR+C (a=0.1, b=0.5)":    (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=0.5,
                                                                    use_clustering=True),                                      False),
            "NetworkVAR+C (a=0.5, b=0.5)":    (NetworkVARRegressor(stage2_alpha=0.5,  correction_bound=0.5,
                                                                    use_clustering=True),                                      False),
            "NetworkVAR+C (a=0.1, b=1.0)":    (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=1.0,
                                                                    use_clustering=True),                                      False),
            "NetworkVAR+C (a=0.1, b=None)":   (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=None,
                                                                    use_clustering=True),                                      False),
        })

    def _network_models_sign_split() -> Dict[str, Any]:
        """Network models with sign-split IDW features enabled."""
        return _with_no_outlier_variants({
            "NetHAR-Split (Lasso a=0.20)":     (NetworkHARRegressor(lasso_alpha=0.20, use_sign_split=True),                  False),
            "NetHAR-Split (Lasso a=0.10)":     (NetworkHARRegressor(lasso_alpha=0.10, use_sign_split=True),                  False),
            "NetHAR-Split (Lasso a=0.05)":     (NetworkHARRegressor(lasso_alpha=0.05, use_sign_split=True),                  False),
            "NetHAR-Split (Lasso a=0.01)":     (NetworkHARRegressor(lasso_alpha=0.01, use_sign_split=True),                  False),
            "NetHAR-Split (OLS)":              (NetworkHARRegressor(lasso_alpha=0.0,  use_sign_split=True),                  False),
            "NetHAR-Split (Ridge a=0.01)":     (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.01,
                                                                      use_sign_split=True),                                    False),
            "NetHAR-Split (Ridge a=0.10)":     (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=0.10,
                                                                      use_sign_split=True),                                    False),
            "NetHAR-Split (Ridge a=1.0)":      (NetworkHARRegressor(lasso_alpha=0.0,  ridge_alpha=1.0,
                                                                      use_sign_split=True),                                    False),
            "NetworkVAR-Split (a=0.0, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.0,  correction_bound=0.5,
                                                                      use_sign_split=True),                                    False),
            "NetworkVAR-Split (a=0.0, b=None)": (NetworkVARRegressor(stage2_alpha=0.0, correction_bound=None,
                                                                       use_sign_split=True),                                   False),
            "NetworkVAR-Split (a=0.1, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=0.5,
                                                                      use_sign_split=True),                                    False),
            "NetworkVAR-Split (a=0.1, b=1.0)": (NetworkVARRegressor(stage2_alpha=0.1,  correction_bound=1.0,
                                                                      use_sign_split=True),                                    False),
            "NetworkVAR-Split (a=0.1, b=None)": (NetworkVARRegressor(stage2_alpha=0.1, correction_bound=None,
                                                                       use_sign_split=True),                                   False),
        })

    def _network_models_sign_split_clustering() -> Dict[str, Any]:
        """Sign-split network models with clustering features enabled."""
        return _with_no_outlier_variants({
            "NetHAR+CSplit (Lasso a=0.20)":    (NetworkHARRegressor(lasso_alpha=0.20, use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetHAR+CSplit (Lasso a=0.01)":    (NetworkHARRegressor(lasso_alpha=0.01, use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetHAR+CSplit (Lasso a=0.05)":    (NetworkHARRegressor(lasso_alpha=0.05, use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetHAR+CSplit (OLS)":             (NetworkHARRegressor(lasso_alpha=0.0, use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetHAR+CSplit (Lasso a=0.1)":    (NetworkHARRegressor(lasso_alpha=0.1, use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetHAR+CSplit (Ridge a=0.01)":    (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=0.01,
                                                                      use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetHAR+CSplit (Ridge a=0.10)":    (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=0.10,
                                                                      use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetHAR+CSplit (Ridge a=1.0)":     (NetworkHARRegressor(lasso_alpha=0.0, ridge_alpha=1.0,
                                                                      use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetworkVAR+CSplit (a=0.0,b=0.5)": (NetworkVARRegressor(stage2_alpha=0.0, correction_bound=0.5,
                                                                      use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetworkVAR+CSplit (a=0.0,b=None)": (NetworkVARRegressor(stage2_alpha=0.0, correction_bound=None,
                                                                       use_clustering=True,
                                                                       use_sign_split=True),                                   False),
            "NetworkVAR+CSplit (a=0.1,b=0.5)": (NetworkVARRegressor(stage2_alpha=0.1, correction_bound=0.5,
                                                                      use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetworkVAR+CSplit (a=0.1,b=1.0)": (NetworkVARRegressor(stage2_alpha=0.1, correction_bound=1.0,
                                                                      use_clustering=True,
                                                                      use_sign_split=True),                                    False),
            "NetworkVAR+CSplit (a=0.1,b=None)": (NetworkVARRegressor(stage2_alpha=0.1, correction_bound=None,
                                                                       use_clustering=True,
                                                                       use_sign_split=True),                                   False),
        })

    def _learned_weight_models(k_val: int) -> Dict[str, Any]:
        """Learned m×k weight-matrix models for a given k."""
        models: Dict[str, Any] = {}
        # m can be 1..k-1; SVD handles any m < k
        for m_val in range(1, k_val):
            models[f"LearnedW (m={m_val}, Ridge a=0.01)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=0.01), False)
            models[f"LearnedW (m={m_val}, Ridge a=1.0)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=1.0), False)
            models[f"LearnedW (m={m_val}, Ridge a=0.1)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=0.1), False)
            models[f"LearnedW (m={m_val}, Lasso a=0.01)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, lasso_alpha=0.01), False)
            models[f"LearnedW (m={m_val}, Lasso a=0.05)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, lasso_alpha=0.05), False)
        return _with_no_outlier_variants(models)

    def _learned_weight_clustering_models(k_val: int) -> Dict[str, Any]:
        """Learned m×k weight-matrix models with clustering features."""
        models: Dict[str, Any] = {}
        for m_val in range(1, k_val):
            models[f"LearnedW+C (m={m_val}, Ridge a=0.01)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=0.01, use_clustering=True), False)
            models[f"LearnedW+C (m={m_val}, Ridge a=1.0)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=1.0, use_clustering=True), False)
            models[f"LearnedW+C (m={m_val}, Ridge a=0.1)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, alpha=0.1, use_clustering=True), False)
            models[f"LearnedW+C (m={m_val}, Lasso a=0.01)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, lasso_alpha=0.01, use_clustering=True), False)
            models[f"LearnedW+C (m={m_val}, Lasso a=0.05)"] = (
                LearnedWeightNetworkHARRegressor(k=k_val, m=m_val, lasso_alpha=0.05, use_clustering=True), False)
        return _with_no_outlier_variants(models)

    # ── Run baselines ────────────────────────────────────────────────────────
    print(f"\nRunning baseline models on {len(tickers)} tickers...")
    metrics_df, pred_store, all_params = cross_val_multi(
        data_dict,
        baseline_catalogue,
        tickers,
        n_splits=2,
        sample_tickers=SAMPLE_TICKERS,
        save_params=True,
    )

    # ── Run squared-correlation network models for each k ────────────────────
    all_net_metrics: list = []
    print("\nRunning squared-correlation network models (k=1..5)...")
    for k_val in KNN_VALUES:
        net_catalogue: Dict[str, Dict[str, Any]] = {
            f"Network [k={k_val}]": _network_models()
        }
        dd_net = data_dicts_net[k_val]
        net_tickers = list(dd_net.keys())
        metrics_k, pred_store_k, params_k = cross_val_multi(
            dd_net,
            net_catalogue,
            net_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_k)
        all_params.extend(params_k)
        for t in pred_store_k:
            new_cols = [c for c in pred_store_k[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_k[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_k[t]

    # ── Run partial-correlation network models for each k ─────────────────────
    print("\nRunning partial-correlation network models (k=1..5)...")
    for k_val in KNN_VALUES:
        pcorr_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr Network [k={k_val}]": _network_models()
        }
        dd_pc = data_dicts_pcorr[k_val]
        pc_tickers = list(dd_pc.keys())
        metrics_pk, pred_store_pk, params_pk = cross_val_multi(
            dd_pc,
            pcorr_catalogue,
            pc_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_pk)
        all_params.extend(params_pk)
        for t in pred_store_pk:
            new_cols = [c for c in pred_store_pk[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_pk[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_pk[t]

    # ── Run exp-kernel SqCorr network models for each k ──────────────────────
    print("\nRunning exp-kernel network models (k=1..5)...")
    for k_val in KNN_VALUES:
        exp_catalogue: Dict[str, Dict[str, Any]] = {
            f"ExpKernel [k={k_val}]": _network_models()
        }
        dd_exp = data_dicts_exp[k_val]
        exp_tickers = list(dd_exp.keys())
        metrics_ek, pred_store_ek, params_ek = cross_val_multi(
            dd_exp,
            exp_catalogue,
            exp_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_ek)
        all_params.extend(params_ek)
        for t in pred_store_ek:
            new_cols = [c for c in pred_store_ek[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_ek[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_ek[t]

    # ── Run clustering-feature network models (standard inv-IDW) ─────────────
    print("\nRunning clustering-feature network models (k=1..5)...")
    for k_val in KNN_VALUES:
        clust_catalogue: Dict[str, Dict[str, Any]] = {
            f"Clustering [k={k_val}]": _network_models_clustering()
        }
        dd_sq = data_dicts_net[k_val]
        sq_tickers = list(dd_sq.keys())
        metrics_cl, pred_store_cl, params_cl = cross_val_multi(
            dd_sq,
            clust_catalogue,
            sq_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_cl)
        all_params.extend(params_cl)
        for t in pred_store_cl:
            new_cols = [c for c in pred_store_cl[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_cl[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_cl[t]

    # ── Run mutual-information network models ────────────────────────────────
    print("\nRunning mutual-information network models (k=1..5)...")
    for k_val in KNN_VALUES:
        mi_catalogue: Dict[str, Dict[str, Any]] = {
            f"MI Network [k={k_val}]": _network_models()
        }
        dd_mi = data_dicts_mi[k_val]
        mi_tickers = list(dd_mi.keys())
        metrics_mi, pred_store_mi, params_mi = cross_val_multi(
            dd_mi,
            mi_catalogue,
            mi_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_mi)
        all_params.extend(params_mi)
        for t in pred_store_mi:
            new_cols = [c for c in pred_store_mi[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_mi[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_mi[t]

    # ── Run exp-kernel + clustering-feature network models ───────────────────
    print("\nRunning exp-kernel + clustering-feature network models (k=1..5)...")
    for k_val in KNN_VALUES:
        expc_catalogue: Dict[str, Dict[str, Any]] = {
            f"Exp+Clustering [k={k_val}]": _network_models_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        expc_tickers = list(dd_exp.keys())
        metrics_ec, pred_store_ec, params_ec = cross_val_multi(
            dd_exp,
            expc_catalogue,
            expc_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_ec)
        all_params.extend(params_ec)
        for t in pred_store_ec:
            new_cols = [c for c in pred_store_ec[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_ec[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_ec[t]

    # ── Run sign-split feature benchmark models ─────────────────────────────
    print("\nRunning sign-split feature benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        split_catalogue: Dict[str, Dict[str, Any]] = {
            f"SplitFeatures [k={k_val}]": _network_models_sign_split()
        }
        dd_sq = data_dicts_net[k_val]
        split_tickers = list(dd_sq.keys())
        metrics_sp, pred_store_sp, params_sp = cross_val_multi(
            dd_sq,
            split_catalogue,
            split_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_sp)
        all_params.extend(params_sp)
        for t in pred_store_sp:
            new_cols = [c for c in pred_store_sp[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_sp[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_sp[t]

    print("\nRunning sign-split + clustering benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        splitc_catalogue: Dict[str, Dict[str, Any]] = {
            f"Split+Clustering [k={k_val}]": _network_models_sign_split_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        splitc_tickers = list(dd_exp.keys())
        metrics_sc, pred_store_sc, params_sc = cross_val_multi(
            dd_exp,
            splitc_catalogue,
            splitc_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_sc)
        all_params.extend(params_sc)
        for t in pred_store_sc:
            new_cols = [c for c in pred_store_sc[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_sc[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_sc[t]

    print("\nRunning PCorr sign-split benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        pcorr_split_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr Split [k={k_val}]": _network_models_sign_split()
        }
        dd_pc = data_dicts_pcorr[k_val]
        pcorr_split_tickers = list(dd_pc.keys())
        metrics_ps, pred_store_ps, params_ps = cross_val_multi(
            dd_pc,
            pcorr_split_catalogue,
            pcorr_split_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_ps)
        all_params.extend(params_ps)
        for t in pred_store_ps:
            new_cols = [c for c in pred_store_ps[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_ps[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_ps[t]

    print("\nRunning exp-kernel sign-split benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        exp_split_catalogue: Dict[str, Dict[str, Any]] = {
            f"ExpKernel Split [k={k_val}]": _network_models_sign_split()
        }
        dd_exp = data_dicts_exp[k_val]
        exp_split_tickers = list(dd_exp.keys())
        metrics_es, pred_store_es, params_es = cross_val_multi(
            dd_exp,
            exp_split_catalogue,
            exp_split_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_es)
        all_params.extend(params_es)
        for t in pred_store_es:
            new_cols = [c for c in pred_store_es[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_es[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_es[t]

    print("\nRunning MI sign-split benchmarks (k=1..5)...")
    for k_val in KNN_VALUES:
        mi_split_catalogue: Dict[str, Dict[str, Any]] = {
            f"MI Split [k={k_val}]": _network_models_sign_split()
        }
        dd_mi = data_dicts_mi[k_val]
        mi_split_tickers = list(dd_mi.keys())
        metrics_ms, pred_store_ms, params_ms = cross_val_multi(
            dd_mi,
            mi_split_catalogue,
            mi_split_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_ms)
        all_params.extend(params_ms)
        for t in pred_store_ms:
            new_cols = [c for c in pred_store_ms[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_ms[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_ms[t]

    # ── Run learned-weight models (SqCorr data) ─────────────────────────────
    print("\nRunning learned-weight models (k=1..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m (need m < k)
        lw_catalogue: Dict[str, Dict[str, Any]] = {
            f"LearnedWeight [k={k_val}]": _learned_weight_models(k_val)
        }
        dd_sq = data_dicts_net[k_val]
        lw_tickers = list(dd_sq.keys())
        metrics_lw, pred_store_lw, params_lw = cross_val_multi(
            dd_sq,
            lw_catalogue,
            lw_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_lw)
        all_params.extend(params_lw)
        for t in pred_store_lw:
            new_cols = [c for c in pred_store_lw[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_lw[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_lw[t]

    # ── Run learned-weight models (PCorr data) ───────────────────────────────
    print("\nRunning learned-weight models (PCorr, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lw_pc_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr LearnedWeight [k={k_val}]": _learned_weight_models(k_val)
        }
        dd_pc = data_dicts_pcorr[k_val]
        lw_pc_tickers = list(dd_pc.keys())
        metrics_lwp, pred_store_lwp, params_lwp = cross_val_multi(
            dd_pc,
            lw_pc_catalogue,
            lw_pc_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_lwp)
        all_params.extend(params_lwp)
        for t in pred_store_lwp:
            new_cols = [c for c in pred_store_lwp[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_lwp[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_lwp[t]

    # ── Run learned-weight models (MI data) ──────────────────────────────────
    print("\nRunning learned-weight models (MI, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lw_mi_catalogue: Dict[str, Dict[str, Any]] = {
            f"MI LearnedWeight [k={k_val}]": _learned_weight_models(k_val)
        }
        dd_mi = data_dicts_mi[k_val]
        lw_mi_tickers = list(dd_mi.keys())
        metrics_lwm, pred_store_lwm, params_lwm = cross_val_multi(
            dd_mi,
            lw_mi_catalogue,
            lw_mi_tickers,
            n_splits=2,
            sample_tickers=SAMPLE_TICKERS,
            save_params=True,
        )
        all_net_metrics.append(metrics_lwm)
        all_params.extend(params_lwm)
        for t in pred_store_lwm:
            new_cols = [c for c in pred_store_lwm[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_lwm[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_lwm[t]

    # ── Run learned-weight + clustering models (SqCorr data) ─────────────────
    print("\nRunning learned-weight + clustering models (k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lwc_catalogue: Dict[str, Dict[str, Any]] = {
            f"LW+Clustering [k={k_val}]": _learned_weight_clustering_models(k_val)
        }
        dd_sq = data_dicts_net[k_val]
        lwc_tickers = list(dd_sq.keys())
        metrics_lwc, pred_store_lwc, params_lwc = cross_val_multi(
            dd_sq, lwc_catalogue, lwc_tickers,
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lwc)
        all_params.extend(params_lwc)
        for t in pred_store_lwc:
            new_cols = [c for c in pred_store_lwc[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_lwc[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_lwc[t]

    # ── Run learned-weight + clustering (PCorr data) ─────────────────────────
    print("\nRunning learned-weight + clustering (PCorr, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lwc_pc_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr LW+Clustering [k={k_val}]": _learned_weight_clustering_models(k_val)
        }
        dd_pc = data_dicts_pcorr[k_val]
        lwc_pc_tickers = list(dd_pc.keys())
        metrics_lwcp, pred_store_lwcp, params_lwcp = cross_val_multi(
            dd_pc, lwc_pc_catalogue, lwc_pc_tickers,
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lwcp)
        all_params.extend(params_lwcp)
        for t in pred_store_lwcp:
            new_cols = [c for c in pred_store_lwcp[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_lwcp[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_lwcp[t]

    # ── Run learned-weight + clustering (MI data) ────────────────────────────
    print("\nRunning learned-weight + clustering (MI, k=2..5)...")
    for k_val in KNN_VALUES:
        if k_val < 2:
            continue  # k=1 → no valid m
        lwc_mi_catalogue: Dict[str, Dict[str, Any]] = {
            f"MI LW+Clustering [k={k_val}]": _learned_weight_clustering_models(k_val)
        }
        dd_mi = data_dicts_mi[k_val]
        lwc_mi_tickers = list(dd_mi.keys())
        metrics_lwcm, pred_store_lwcm, params_lwcm = cross_val_multi(
            dd_mi, lwc_mi_catalogue, lwc_mi_tickers,
            n_splits=2, sample_tickers=SAMPLE_TICKERS, save_params=True,
        )
        all_net_metrics.append(metrics_lwcm)
        all_params.extend(params_lwcm)
        for t in pred_store_lwcm:
            new_cols = [c for c in pred_store_lwcm[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_lwcm[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_lwcm[t]

    metrics_df = pd.concat([metrics_df] + all_net_metrics, ignore_index=True)

    # Save the full (un-coalesced) results for later analysis
    save_results(metrics_df, summarize_benchmarks(metrics_df), RESULTS_DIR)
    save_model_params(all_params, RESULTS_DIR, "regression_model_params.json")
    save_prediction_store(pred_store, RESULTS_DIR)

    # Coalesce k-variants into super-categories for printing/plotting
    metrics_coalesced = coalesce_categories(metrics_df)

    # ── Validation / Test split reporting ─────────────────────────────────────
    # Fold 1 = validation (earlier year), Fold 0 = test (most recent year).
    # Select best model per category on the validation fold, then report
    # performance of those models on the held-out test fold.
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

    # ── Best-model selection on validation, reported on test ─────────────────
    best_models = select_best_on_validation(metrics_coalesced, val_fold=1)
    print("\n" + "=" * 80)
    print("BEST MODEL PER CATEGORY  (chosen on validation fold)")
    print("=" * 80)
    print(best_models.to_string(index=False))

    print_best_test_summary(metrics_coalesced, best_models, test_fold=0)

    # ── Legacy full-summary (both folds averaged) for reference ──────────────
    summary = summarize_benchmarks(metrics_coalesced)
    print_summary(summary, title="Full Summary — both folds (all stocks)")
    print_compact_leaderboard(summary)
    print_summary_excluding_outliers(metrics_coalesced, r2_threshold=-1.0)
    print_wilcoxon_best_network_vs_baseline(metrics_coalesced)
    print_per_ticker_tables(metrics_coalesced, SAMPLE_TICKERS)

    # Plotting intentionally omitted in this evaluation script.


def sanity_main():
    import os
    """
    Quick smoke test: one model per category, single k-value, 2 tickers.
    Runs in ~1 minute instead of hours.  Good for verifying plumbing after
    code changes before committing to a full run.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data import get_data_for_har
    from models.correlation_network import SquaredCorrelationNetwork, PartialCorrelationNetwork

    SAMPLE = ["AAPL", "MSFT"]
    K_VAL = 3
    graph_n_jobs = max(1, min(4, (os.cpu_count() or 1) - 1))

    print("=== SANITY CHECK ===")
    print("Loading data (top 35 tickers)...")
    data_dict = get_data_for_har(35)
    tickers = [t for t in data_dict if t in SAMPLE] or list(data_dict.keys())[:2]
    print(f"  Tickers for test: {tickers}")

    # ── Graph builds ──────────────────────────────────────────────────────────
    print(f"\nBuilding SquaredCorrelationNetwork (k={K_VAL}, inv-kernel)...")
    net_sq = SquaredCorrelationNetwork(
        window=60, step=1, save_step=5, n_jobs=graph_n_jobs, graph_type="knn", k=K_VAL,
        feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
    )
    dd_sq = net_sq.fit_transform(data_dict)
    print(f"  {net_sq.n_all_snapshots_} total graphs, {net_sq.n_snapshots_} saved.")

    print(f"\nBuilding PartialCorrelationNetwork (k={K_VAL})...")
    net_pc = PartialCorrelationNetwork(
        window=60, step=1, save_step=5, n_jobs=graph_n_jobs, graph_type="knn", k=K_VAL, shrinkage=0.1,
        feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
    )
    dd_pc = net_pc.fit_transform(data_dict)
    print(f"  {net_pc.n_all_snapshots_} total graphs, {net_pc.n_snapshots_} saved.")

    print(f"\nBuilding SquaredCorrelationNetwork (k={K_VAL}, exp-kernel)...")
    net_exp = SquaredCorrelationNetwork(
        window=60, step=1, save_step=5, n_jobs=graph_n_jobs, graph_type="knn", k=K_VAL,
        feature_cols=["log_RV1", "log_RV5", "log_RV22", "Returns"],
        idw_kernel="exp", exp_lambda=5.0,
    )
    dd_exp = net_exp.fit_transform(data_dict)
    print(f"  {net_exp.n_all_snapshots_} total graphs, {net_exp.n_snapshots_} saved.")

    # ── Minimal catalogue: one model per category ─────────────────────────────
    catalogue: Dict[str, Dict[str, Any]] = {
        "HAR":   {"HAR-Extended": (HARExtendedLogRegressor(), False)},
        "ARIMA": {"ARIMA(1,1,1)": (ARIMALogY(order=(1, 1, 1)), False)},
        "GARCH": {
            "GARCH(1,1)": (GARCHWeeklyRV(p=1, q=1, horizon=5), False),
            "DCC-GARCH(1,1)": (DCCGARCHWeeklyRV(p=1, q=1, horizon=5), False),
        },
    }

    print(f"\nRunning baseline catalogue ({len(tickers)} tickers)...")
    metrics_df, pred_store = cross_val_multi(
        data_dict, catalogue, tickers,
        n_splits=1, sample_tickers=SAMPLE,
    )

    # Network models
    all_extra: list = []

    # SqCorr (inv kernel, no clustering)
    net_cat: Dict[str, Dict[str, Any]] = {
        f"Network [k={K_VAL}]": {
            "NetHAR (Lasso a=0.05)": (NetworkHARRegressor(lasso_alpha=0.05), False),
        }
    }
    print(f"\nRunning Network [k={K_VAL}] (squared-corr, inv)...")
    m_sq, ps_sq = cross_val_multi(
        dd_sq, net_cat, list(dd_sq.keys()),
        n_splits=1, sample_tickers=SAMPLE,
    )
    all_extra.append(m_sq)

    # PCorr
    pc_cat: Dict[str, Dict[str, Any]] = {
        f"PCorr Network [k={K_VAL}]": {
            "NetHAR (Lasso a=0.05)": (NetworkHARRegressor(lasso_alpha=0.05), False),
        }
    }
    print(f"\nRunning PCorr Network [k={K_VAL}]...")
    m_pc, ps_pc = cross_val_multi(
        dd_pc, pc_cat, list(dd_pc.keys()),
        n_splits=1, sample_tickers=SAMPLE,
    )
    all_extra.append(m_pc)

    # Exp kernel
    ek_cat: Dict[str, Dict[str, Any]] = {
        f"ExpKernel [k={K_VAL}]": {
            "NetHAR (Lasso a=0.05)": (NetworkHARRegressor(lasso_alpha=0.05), False),
        }
    }
    print(f"\nRunning ExpKernel [k={K_VAL}]...")
    m_ek, ps_ek = cross_val_multi(
        dd_exp, ek_cat, list(dd_exp.keys()),
        n_splits=1, sample_tickers=SAMPLE,
    )
    all_extra.append(m_ek)

    # Clustering (uses SqCorr data with clustering models)
    cl_cat: Dict[str, Dict[str, Any]] = {
        f"Clustering [k={K_VAL}]": {
            "NetHAR+C (Lasso a=0.05)": (NetworkHARRegressor(lasso_alpha=0.05, use_clustering=True), False),
        }
    }
    print(f"\nRunning Clustering [k={K_VAL}]...")
    m_cl, ps_cl = cross_val_multi(
        dd_sq, cl_cat, list(dd_sq.keys()),
        n_splits=1, sample_tickers=SAMPLE,
    )
    all_extra.append(m_cl)

    # Exp + Clustering
    ec_cat: Dict[str, Dict[str, Any]] = {
        f"Exp+Clustering [k={K_VAL}]": {
            "NetHAR+C (Lasso a=0.05)": (NetworkHARRegressor(lasso_alpha=0.05, use_clustering=True), False),
        }
    }
    print(f"\nRunning Exp+Clustering [k={K_VAL}]...")
    m_ec, ps_ec = cross_val_multi(
        dd_exp, ec_cat, list(dd_exp.keys()),
        n_splits=1, sample_tickers=SAMPLE,
    )
    all_extra.append(m_ec)

    # Merge pred_stores
    for ps in [ps_sq, ps_pc, ps_ek, ps_cl, ps_ec]:
        for t in ps:
            cols = [c for c in ps[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(ps[t][cols], how="outer")

    metrics_df = pd.concat([metrics_df] + all_extra, ignore_index=True)

    summary = summarize_benchmarks(metrics_df)
    print("\n=== SANITY RESULTS ===")
    print_summary(summary, title="Sanity Summary")
    print_best_per_category(summary)

    # Quick column-collision check
    for t, df in pred_store.items():
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            print(f"  WARNING: duplicate columns for {t}: {dupes}")
        else:
            print(f"  {t}: {len(df.columns)} columns, no duplicates ✓")

    print("\n=== SANITY CHECK PASSED ===")


def test_dcc(
    ticker: str = "AAPL",
    num_tickers: int = 25,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Run only the DCC-GARCH baseline on a single ticker and print a compact
    smoke-test report.

    Usage
    -----
    python evaluation/cross_val.py test_dcc
    python evaluation/cross_val.py test_dcc MSFT
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data import get_data_for_har

    print("=== DCC-GARCH TEST ===")
    print(f"Loading and preprocessing top {num_tickers} tickers...")
    data_dict = get_data_for_har(num_tickers)
    if not data_dict:
        raise RuntimeError("No preprocessed ticker data was loaded.")

    if ticker not in data_dict:
        fallback = next(iter(data_dict))
        print(f"Ticker {ticker} not found in loaded universe; using {fallback} instead.")
        ticker = fallback

    feat = data_dict[ticker]
    print(f"Using ticker: {ticker} ({len(feat)} rows)")

    fold_iter = expanding_folds(
        feat.index,
        n_splits=1,
        test_size=test_size,
        min_train_size=min_train_size,
    )
    try:
        train_idx, test_idx = next(fold_iter)
    except StopIteration as exc:
        raise RuntimeError(
            f"Not enough data for ticker {ticker} with min_train_size={min_train_size} and test_size={test_size}."
        ) from exc

    X_train = feat.loc[train_idx]
    X_test = feat.loc[test_idx]
    y_train = X_train["log_Y"]
    y_true = feat.loc[test_idx, "Y_fwd"]

    model = DCCGARCHWeeklyRV(p=1, q=1, horizon=5)
    print(
        f"Fitting DCC-GARCH on {len(X_train)} training rows and forecasting {len(X_test)} test rows..."
    )
    model.fit(X_train[model.features], y_train)
    y_pred = model.predict(X_test[model.features])
    metrics = eval_regression(y_true.values, y_pred)

    preview = pd.DataFrame(
        {
            "Y_true": y_true,
            "Y_pred": y_pred,
        },
        index=test_idx,
    )

    print("\nDCC metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    print("\nPrediction preview:")
    print(preview.head(10).round(6).to_string())
    print("\n=== DCC-GARCH TEST PASSED ===")
    return preview, metrics


if __name__ == "__main__":
    main()
    
    # test_dcc(ticker="NFLX")
    # sanity_main()
    # load_and_print_results()