import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.baselines import HARLogRegressor, HARExtendedLogRegressor, ARIMALogY, GARCHWeeklyRV
from models.network_models import NetworkHARRegressor, NetworkVARRegressor
from data.preprocess import remove_outliers
from visualize.utils import coalesce_categories
from visualize.print_results import (
    summarize_benchmarks,
    print_summary,
    print_best_per_category,
    print_per_ticker_tables,
    print_summary_excluding_outliers,
    load_and_print_results,
)


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
) -> "tuple[pd.DataFrame, dict]":
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

                    if t in sample_set:
                        col_key = f"[{category}] {model_name}"
                        pred_df.loc[test_idx, col_key] = y_pred

        if t in sample_set:
            pred_store[t] = pred_df

    metrics_df = pd.DataFrame(rows)
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


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data import get_data_for_har
    from visualize.plot_model_results import (
        plot_ticker_predictions,
        plot_summary_metrics,
        plot_network_degrees,
    )

    SAMPLE_TICKERS = ["AAPL", "TSLA", "GOOG", "META", "MSFT", "NVDA", "NFLX", "AMZN"]
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading and preprocessing data...")
    data_dict = get_data_for_har(100)
    tickers = list(data_dict.keys())

    # ── Offline graph build ──────────────────────────────────────────────────
    # One SquaredCorrelationNetwork per k value, built ONCE on the full dataset.
    # Each snapshot at date t only uses returns r_{t-window+1}...r_t (past only),
    # so fitting on the full history does NOT introduce look-ahead leakage.
    print("\nBuilding correlation-network features offline (k=1, 3, 5)...")
    from models.correlation_network import SquaredCorrelationNetwork, PartialCorrelationNetwork
    KNN_VALUES = [1, 3, 5]
    nets: dict = {}
    data_dicts_net: dict = {}
    for k_val in KNN_VALUES:
        net_k = SquaredCorrelationNetwork(
            window=60,
            step=5,
            graph_type="knn",
            k=k_val,
            feature_cols=["log_RV1", "log_RV5", "log_RV22"],
        )
        data_dicts_net[k_val] = net_k.fit_transform(data_dict)
        nets[k_val] = net_k
        print(f"  [SqCorr] k={k_val}: built {net_k.n_snapshots_} graph snapshots.")
    net = nets[5]   # reference for degree-dynamics plot

    print("\nBuilding partial-correlation network features offline (k=1, 3, 5)...")
    nets_pcorr: dict = {}
    data_dicts_pcorr: dict = {}
    for k_val in KNN_VALUES:
        net_pk = PartialCorrelationNetwork(
            window=60,
            step=5,
            graph_type="knn",
            k=k_val,
            shrinkage=0.1,
            feature_cols=["log_RV1", "log_RV5", "log_RV22"],
        )
        data_dicts_pcorr[k_val] = net_pk.fit_transform(data_dict)
        nets_pcorr[k_val] = net_pk
        print(f"  [PCorr]  k={k_val}: built {net_pk.n_snapshots_} snapshots.")

    # Exp-kernel: same SquaredCorrelationNetwork but with exp(-lambda*d) IDW
    print("\nBuilding exp-kernel correlation-network features offline (k=1, 3, 5)...")
    data_dicts_exp: dict = {}
    for k_val in KNN_VALUES:
        net_exp = SquaredCorrelationNetwork(
            window=60,
            step=5,
            graph_type="knn",
            k=k_val,
            feature_cols=["log_RV1", "log_RV5", "log_RV22"],
            idw_kernel="exp",
            exp_lambda=5.0,
        )
        data_dicts_exp[k_val] = net_exp.fit_transform(data_dict)
        print(f"  [ExpKernel] k={k_val}: built {net_exp.n_snapshots_} snapshots.")

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
        },
    }

    def _network_models() -> Dict[str, Any]:
        """Return fresh network model instances (needed per k-value run)."""
        return {
            "NetHAR (Lasso a=0.10)":     (NetworkHARRegressor(lasso_alpha=0.10),           False),
            "NetHAR (Lasso a=0.05)":     (NetworkHARRegressor(lasso_alpha=0.05),           False),
            "NetHAR (Lasso a=0.01)":     (NetworkHARRegressor(lasso_alpha=0.01),           False),
            "NetHAR (OLS)":              (NetworkHARRegressor(lasso_alpha=0.0),             False),
            "NetworkVAR (a=0.1, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.1,
                                                               correction_bound=0.5),      False),
            "NetworkVAR (a=0.5, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.5,
                                                               correction_bound=0.5),      False),
            "NetworkVAR (a=0.1, b=1.0)": (NetworkVARRegressor(stage2_alpha=0.1,
                                                               correction_bound=1.0),      False),
        }

    def _network_models_clustering() -> Dict[str, Any]:
        """Network models with clustering features enabled."""
        return {
            "NetHAR+C (Lasso a=0.10)":     (NetworkHARRegressor(lasso_alpha=0.10, use_clustering=True), False),
            "NetHAR+C (Lasso a=0.05)":     (NetworkHARRegressor(lasso_alpha=0.05, use_clustering=True), False),
            "NetHAR+C (Lasso a=0.01)":     (NetworkHARRegressor(lasso_alpha=0.01, use_clustering=True), False),
            "NetHAR+C (OLS)":              (NetworkHARRegressor(lasso_alpha=0.0,  use_clustering=True), False),
            "NetworkVAR+C (a=0.1, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.1,
                                                                  correction_bound=0.5,
                                                                  use_clustering=True), False),
            "NetworkVAR+C (a=0.5, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.5,
                                                                  correction_bound=0.5,
                                                                  use_clustering=True), False),
            "NetworkVAR+C (a=0.1, b=1.0)": (NetworkVARRegressor(stage2_alpha=0.1,
                                                                  correction_bound=1.0,
                                                                  use_clustering=True), False),
        }

    # ── Run baselines ────────────────────────────────────────────────────────
    print(f"\nRunning baseline models on {len(tickers)} tickers...")
    metrics_df, pred_store = run_benchmarks_multi_fold(
        data_dict,
        baseline_catalogue,
        tickers,
        n_splits=1,
        sample_tickers=SAMPLE_TICKERS,
    )

    # ── Run squared-correlation network models for each k ────────────────────
    all_net_metrics: list = []
    print("\nRunning squared-correlation network models (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        net_catalogue: Dict[str, Dict[str, Any]] = {
            f"Network [k={k_val}]": _network_models()
        }
        dd_net = data_dicts_net[k_val]
        net_tickers = list(dd_net.keys())
        metrics_k, pred_store_k = run_benchmarks_multi_fold(
            dd_net,
            net_catalogue,
            net_tickers,
            n_splits=1,
            sample_tickers=SAMPLE_TICKERS,
        )
        all_net_metrics.append(metrics_k)
        for t in pred_store_k:
            new_cols = [c for c in pred_store_k[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_k[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_k[t]

    # ── Run partial-correlation network models for each k ─────────────────────
    print("\nRunning partial-correlation network models (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        pcorr_catalogue: Dict[str, Dict[str, Any]] = {
            f"PCorr Network [k={k_val}]": _network_models()
        }
        dd_pc = data_dicts_pcorr[k_val]
        pc_tickers = list(dd_pc.keys())
        metrics_pk, pred_store_pk = run_benchmarks_multi_fold(
            dd_pc,
            pcorr_catalogue,
            pc_tickers,
            n_splits=1,
            sample_tickers=SAMPLE_TICKERS,
        )
        all_net_metrics.append(metrics_pk)
        for t in pred_store_pk:
            new_cols = [c for c in pred_store_pk[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_pk[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_pk[t]

    # ── Run exp-kernel SqCorr network models for each k ──────────────────────
    print("\nRunning exp-kernel network models (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        exp_catalogue: Dict[str, Dict[str, Any]] = {
            f"ExpKernel [k={k_val}]": _network_models()
        }
        dd_exp = data_dicts_exp[k_val]
        exp_tickers = list(dd_exp.keys())
        metrics_ek, pred_store_ek = run_benchmarks_multi_fold(
            dd_exp,
            exp_catalogue,
            exp_tickers,
            n_splits=1,
            sample_tickers=SAMPLE_TICKERS,
        )
        all_net_metrics.append(metrics_ek)
        for t in pred_store_ek:
            new_cols = [c for c in pred_store_ek[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_ek[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_ek[t]

    # ── Run clustering-feature network models (standard inv-IDW) ─────────────
    print("\nRunning clustering-feature network models (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        clust_catalogue: Dict[str, Dict[str, Any]] = {
            f"Clustering [k={k_val}]": _network_models_clustering()
        }
        dd_sq = data_dicts_net[k_val]
        sq_tickers = list(dd_sq.keys())
        metrics_cl, pred_store_cl = run_benchmarks_multi_fold(
            dd_sq,
            clust_catalogue,
            sq_tickers,
            n_splits=1,
            sample_tickers=SAMPLE_TICKERS,
        )
        all_net_metrics.append(metrics_cl)
        for t in pred_store_cl:
            new_cols = [c for c in pred_store_cl[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_cl[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_cl[t]

    # ── Run exp-kernel + clustering-feature network models ───────────────────
    print("\nRunning exp-kernel + clustering-feature network models (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        expc_catalogue: Dict[str, Dict[str, Any]] = {
            f"Exp+Clustering [k={k_val}]": _network_models_clustering()
        }
        dd_exp = data_dicts_exp[k_val]
        expc_tickers = list(dd_exp.keys())
        metrics_ec, pred_store_ec = run_benchmarks_multi_fold(
            dd_exp,
            expc_catalogue,
            expc_tickers,
            n_splits=1,
            sample_tickers=SAMPLE_TICKERS,
        )
        all_net_metrics.append(metrics_ec)
        for t in pred_store_ec:
            new_cols = [c for c in pred_store_ec[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_ec[t][new_cols], how="outer")
            else:
                pred_store[t] = pred_store_ec[t]

    metrics_df = pd.concat([metrics_df] + all_net_metrics, ignore_index=True)

    # Save the full (un-coalesced) results for later analysis
    save_results(metrics_df, summarize_benchmarks(metrics_df), RESULTS_DIR)

    # Coalesce k-variants into super-categories for printing/plotting
    metrics_coalesced = coalesce_categories(metrics_df)
    summary = summarize_benchmarks(metrics_coalesced)

    # ── Detailed printing ─────────────────────────────────────────────────────
    print_summary(summary, title="Full Summary (all stocks)")
    print_best_per_category(summary)
    print_summary_excluding_outliers(metrics_coalesced, r2_threshold=-1.0)
    print_per_ticker_tables(metrics_coalesced, SAMPLE_TICKERS)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_network_degrees(
        net,
        sample_tickers=SAMPLE_TICKERS,
        save_path=str(RESULTS_DIR / "network_degrees.png"),
    )
    plot_summary_metrics(summary, save_path=str(RESULTS_DIR / "summary_metrics.png"))
    plot_ticker_predictions(
        pred_store,
        metrics_coalesced,
        SAMPLE_TICKERS,
        save_dir=str(RESULTS_DIR),
    )


def sanity_main():
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

    print("=== SANITY CHECK ===")
    print("Loading data (top 35 tickers)...")
    data_dict = get_data_for_har(35)
    tickers = [t for t in data_dict if t in SAMPLE] or list(data_dict.keys())[:2]
    print(f"  Tickers for test: {tickers}")

    # ── Graph builds ──────────────────────────────────────────────────────────
    print(f"\nBuilding SquaredCorrelationNetwork (k={K_VAL}, inv-kernel)...")
    net_sq = SquaredCorrelationNetwork(
        window=60, step=5, graph_type="knn", k=K_VAL,
        feature_cols=["log_RV1", "log_RV5", "log_RV22"],
    )
    dd_sq = net_sq.fit_transform(data_dict)
    print(f"  {net_sq.n_snapshots_} snapshots.")

    print(f"\nBuilding PartialCorrelationNetwork (k={K_VAL})...")
    net_pc = PartialCorrelationNetwork(
        window=60, step=5, graph_type="knn", k=K_VAL, shrinkage=0.1,
        feature_cols=["log_RV1", "log_RV5", "log_RV22"],
    )
    dd_pc = net_pc.fit_transform(data_dict)
    print(f"  {net_pc.n_snapshots_} snapshots.")

    print(f"\nBuilding SquaredCorrelationNetwork (k={K_VAL}, exp-kernel)...")
    net_exp = SquaredCorrelationNetwork(
        window=60, step=5, graph_type="knn", k=K_VAL,
        feature_cols=["log_RV1", "log_RV5", "log_RV22"],
        idw_kernel="exp", exp_lambda=5.0,
    )
    dd_exp = net_exp.fit_transform(data_dict)
    print(f"  {net_exp.n_snapshots_} snapshots.")

    # ── Minimal catalogue: one model per category ─────────────────────────────
    catalogue: Dict[str, Dict[str, Any]] = {
        "HAR":   {"HAR-Extended": (HARExtendedLogRegressor(), False)},
        "ARIMA": {"ARIMA(1,1,1)": (ARIMALogY(order=(1, 1, 1)), False)},
        "GARCH": {"GARCH(1,1)":   (GARCHWeeklyRV(p=1, q=1, horizon=5), False)},
    }

    print(f"\nRunning baseline catalogue ({len(tickers)} tickers)...")
    metrics_df, pred_store = run_benchmarks_multi_fold(
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
    m_sq, ps_sq = run_benchmarks_multi_fold(
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
    m_pc, ps_pc = run_benchmarks_multi_fold(
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
    m_ek, ps_ek = run_benchmarks_multi_fold(
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
    m_cl, ps_cl = run_benchmarks_multi_fold(
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
    m_ec, ps_ec = run_benchmarks_multi_fold(
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


if __name__ == "__main__":
    main()
    # sanity_main()
    # load_and_print_results()