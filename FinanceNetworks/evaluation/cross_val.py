import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.baselines import HARLogRegressor, HARExtendedLogRegressor, ARIMALogY, GARCHWeeklyRV
from models.network_models import NetworkHARRegressor, NetworkVARRegressor
from data.preprocess import remove_outliers


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
    model_dict: "dict[str, Any]",
    tickers: "list[str]",
    n_splits: int = 1,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
    sample_tickers: "list[str] | None" = None,
) -> "tuple[pd.DataFrame, dict]":
    """
    Run expanding-window cross-validation for every model in *model_dict*
    over all *tickers*.

    Parameters
    ----------
    data_dict     : {ticker: DataFrame} with HAR features, log_Y, Y_fwd, and
                    any columns required by the models (e.g. Returns for GARCH).
    model_dict    : {display_name: model_instance} or
                    {display_name: (model_instance, remove_outliers_bool)}.
    n_splits      : Number of expanding-window folds.  Use 1 (default) for a
                    single hold-out evaluation -- easiest to interpret.
    sample_tickers: Store per-row predictions for these tickers so they can
                    be plotted after the run.

    Returns
    -------
    metrics_df : Long-form DataFrame with columns
                 [Ticker, Model, Fold, R2, RMSE, MAE, R2_log, RMSE_log, MAE_log].
    pred_store : {ticker: DataFrame} with aligned predictions for each ticker
                 listed in sample_tickers.
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

            for model_name, model_entry in model_dict.items():
                if isinstance(model_entry, tuple):
                    model_template, do_remove_outliers = model_entry
                else:
                    model_template, do_remove_outliers = model_entry, False

                model = copy.deepcopy(model_template)
                X_train_fit = remove_outliers(X_train) if do_remove_outliers else X_train
                y_pred = fit_predict_model(model, X_train_fit, X_test)
                m = eval_regression(y_true.loc[test_idx].values, y_pred)
                rows.append({"Ticker": t, "Model": model_name, "Fold": fold_id, **m})

                if t in sample_set:
                    pred_df.loc[test_idx, model_name] = y_pred

        if t in sample_set:
            pred_store[t] = pred_df

    metrics_df = pd.DataFrame(rows)
    return metrics_df, pred_store


def summarize_benchmarks(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics first per ticker, then across tickers."""
    per_ticker = (
        metrics_df.groupby(["Ticker", "Model"])
        .agg(
            R2=("R2", "mean"),
            RMSE=("RMSE", "mean"),
            MAE=("MAE", "mean"),
            R2_log=("R2_log", "mean"),
            RMSE_log=("RMSE_log", "mean"),
            MAE_log=("MAE_log", "mean"),
        )
        .reset_index()
    )

    summary = (
        per_ticker.groupby("Model")
        .agg(
            mean_R2=("R2", "mean"),
            median_R2=("R2", "median"),
            pct_R2_pos=("R2", lambda x: float((x > 0).mean())),
            mean_RMSE=("RMSE", "mean"),
            median_RMSE=("RMSE", "median"),
            mean_R2_log=("R2_log", "mean"),
            median_R2_log=("R2_log", "median"),
            mean_RMSE_log=("RMSE_log", "mean"),
            mean_MAE_log=("MAE_log", "mean"),
            n_tickers=("Ticker", "nunique"),
        )
        .sort_values("median_R2_log", ascending=False)
    )
    return summary


def print_summary(summary: pd.DataFrame) -> None:
    """Print the full summary table without column truncation."""
    with pd.option_context(
        "display.max_columns", None,
        "display.width", None,
        "display.float_format", "{:.4f}".format,
    ):
        print(summary.to_string())


def save_results(
    metrics_df: pd.DataFrame,
    summary: pd.DataFrame,
    results_dir: Path,
) -> None:
    """
    Persist cross-validation outputs to disk.

    Files written
    -------------
    results_bench.json   : fold-level metrics, one record per (Ticker, Model, Fold).
    summary_bench.json   : model-level summary aggregated across tickers, one
                           record per Model, sorted by median_R2_log descending.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Fold-level raw metrics
    raw_path = results_dir / "results_bench.json"
    metrics_df.to_json(raw_path, orient="records", indent=2)
    print(f"Fold-level metrics  → {raw_path}")

    # Model-level summary (reset_index so Model is a column, not the index)
    summary_path = results_dir / "summary_bench.json"
    summary.reset_index().to_json(summary_path, orient="records", indent=2)
    print(f"Summary metrics     → {summary_path}")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data import get_data_for_har
    from visualize.plot_model_results import (
        plot_ticker_predictions,
        plot_summary_metrics,
        plot_network_degrees,
    )

    SAMPLE_TICKERS = ["AAPL", "TSLA", "GOOG", "META", "MSFT"]
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading and preprocessing data...")
    data_dict = get_data_for_har(100)
    tickers = list(data_dict.keys())

    # ── Offline graph build (no leakage -- each snapshot uses only past returns)
    # Build one SquaredCorrelationNetwork per k value so we can evaluate how
    # graph sparsity (k=1 sparse, k=3 medium, k=5 dense) affects forecast quality.
    print("\nBuilding correlation-network features offline (k=1, 3, 5)...")
    from models.correlation_network import SquaredCorrelationNetwork
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
        print(f"  k={k_val}: built {net_k.n_snapshots_} graph snapshots.")
    # Keep a reference to k=5 for degree-dynamics plot (most connected / richest graph)
    net = nets[5]

    # ── Model catalogue ─────────────────────────────────────────────────────
    # Each entry: (model_instance, strip_outliers_from_train)
    # OLS / Ridge / Lasso HAR variants
    # ARIMA orders from parsimonious to richer
    # GARCH with varying p and q
    model_dict = {
        # ── HAR / HAR-Extended (OLS) ─────────────────────────────────────
        "HAR":                        (HARLogRegressor(),                               False),
        "HAR (no outliers)":          (HARLogRegressor(),                               True),
        "HAR-Extended":               (HARExtendedLogRegressor(),                       False),
        "HAR-Extended (no outliers)": (HARExtendedLogRegressor(),                       True),
        # ── HAR Lasso ────────────────────────────────────────────────────
        "HAR-Lasso (a=0.01)":         (HARLogRegressor(lasso_alpha=0.01),               False),
        "HAR-Lasso (a=0.1)":          (HARLogRegressor(lasso_alpha=0.1),                False),
        "HAR-Ext-Lasso (a=0.01)":     (HARExtendedLogRegressor(lasso_alpha=0.01),       False),
        "HAR-Ext-Lasso (a=0.1)":      (HARExtendedLogRegressor(lasso_alpha=0.1),        False),
        # ── ARIMA ────────────────────────────────────────────────────────
        "ARIMA(1,0,1)":               (ARIMALogY(order=(1, 0, 1)),                      False),
        "ARIMA(2,0,1)":               (ARIMALogY(order=(2, 0, 1)),                      False),
        "ARIMA(1,0,2)":               (ARIMALogY(order=(1, 0, 2)),                      False),
        "ARIMA(2,0,2)":               (ARIMALogY(order=(2, 0, 2)),                      False),
        # ── GARCH ────────────────────────────────────────────────────────
        "GARCH(1,1)":                 (GARCHWeeklyRV(p=1, q=1, horizon=5),              False),
        "GARCH(2,1)":                 (GARCHWeeklyRV(p=2, q=1, horizon=5),              False),
        "GARCH(1,2)":                 (GARCHWeeklyRV(p=1, q=2, horizon=5),              False),
        "GARCH(2,2)":                 (GARCHWeeklyRV(p=2, q=2, horizon=5),              False),
    }

    # ── Network models use data_dict_net (HAR features + net_* columns) ─
    # The same model templates are evaluated for each k, with the k value
    # appended to the model name so results are distinguishable in the summary.
    def _network_model_templates():
        return {
            "NetHAR (Lasso a=0.05)":     (NetworkHARRegressor(lasso_alpha=0.05),        False),
            "NetHAR (Lasso a=0.01)":     (NetworkHARRegressor(lasso_alpha=0.01),        False),
            "NetHAR (OLS)":              (NetworkHARRegressor(lasso_alpha=0.0),          False),
            "NetworkVAR (a=0.1, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.1,
                                                               correction_bound=0.5),   False),
            "NetworkVAR (a=0.5, b=0.5)": (NetworkVARRegressor(stage2_alpha=0.5,
                                                               correction_bound=0.5),   False),
            "NetworkVAR (a=0.1, b=1.0)": (NetworkVARRegressor(stage2_alpha=0.1,
                                                               correction_bound=1.0),   False),
        }

    print(f"Running single-fold cross-validation on {len(tickers)} tickers...")
    metrics_df, pred_store = run_benchmarks_multi_fold(
        data_dict,
        model_dict,
        tickers,
        n_splits=1,
        sample_tickers=SAMPLE_TICKERS,
    )

    all_net_metrics = []
    print("\nRunning network models (k=1, 3, 5)...")
    for k_val in KNN_VALUES:
        # Tag each model name with the k value for traceability in the summary
        tagged = {
            f"{name} [k={k_val}]": entry
            for name, entry in _network_model_templates().items()
        }
        dd_net = data_dicts_net[k_val]
        net_tickers = list(dd_net.keys())
        metrics_k, pred_store_k = run_benchmarks_multi_fold(
            dd_net,
            tagged,
            net_tickers,
            n_splits=1,
            sample_tickers=SAMPLE_TICKERS,
        )
        all_net_metrics.append(metrics_k)

        # Merge predictions for sample tickers
        for t in pred_store_k:
            net_cols = [c for c in pred_store_k[t].columns if c != "Y_true"]
            if t in pred_store:
                pred_store[t] = pred_store[t].join(pred_store_k[t][net_cols], how="outer")
            else:
                pred_store[t] = pred_store_k[t]

    metrics_df = pd.concat([metrics_df] + all_net_metrics, ignore_index=True)

    summary = summarize_benchmarks(metrics_df)

    print("\nSummary (all columns):")
    print_summary(summary)

    save_results(metrics_df, summary, RESULTS_DIR)

    # ── Plots ────────────────────────────────────────────────────────────────
    # Degree dynamics for the k=5 (richest) network
    plot_network_degrees(
        net,
        sample_tickers=SAMPLE_TICKERS,
        save_path=str(RESULTS_DIR / "network_degrees.png"),
    )
    plot_summary_metrics(summary, save_path=str(RESULTS_DIR / "summary_metrics.png"))
    plot_ticker_predictions(pred_store, SAMPLE_TICKERS, save_dir=str(RESULTS_DIR))


if __name__ == "__main__":
    main()