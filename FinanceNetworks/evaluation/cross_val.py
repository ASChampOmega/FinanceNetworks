import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.baselines import HARLogRegressor, ARIMALogY, GARCHWeeklyRV
from data.preprocess import remove_outliers


def expanding_folds(
    index: "pd.Index",
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
):
    """
    Expanding-window folds with contiguous test blocks.
    Yields (train_idx, test_idx) as index values (timestamps).
    """
    n = len(index)
    if n < min_train_size + test_size:
        raise ValueError(f"Not enough points (n={n}) for min_train_size+test_size.")

    last_test_end = n
    for k in range(n_splits):
        test_end   = last_test_end - k * test_size
        test_start = test_end - test_size
        train_end  = test_start

        if train_end < min_train_size:
            break

        train_idx = index[:train_end]
        test_idx  = index[test_start:test_end]
        yield train_idx, test_idx


def eval_regression(y_true, y_pred, eps: float = 1e-8) -> dict:
    """Evaluate in both original and log scale. Log-scale metrics are more
    stable for heavy-tailed realized-variance targets."""
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
    """Fit model on the training slice and return predictions for the test slice."""
    X_train1 = X_train[model.features]
    X_test1  = X_test[model.features]
    y_train  = X_train["log_Y"]
    model.fit(X_train1, y_train)
    return model.predict(X_test1)


def run_benchmarks_multi_fold(
    data_dict: "dict[str, pd.DataFrame]",
    model_dict: "dict[str, Any]",
    tickers: "list[str]",
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 252 * 5,
    sample_tickers: "list[str] | None" = None,
) -> "tuple[pd.DataFrame, dict]":
    """
    Run expanding-window cross-validation for every model in model_dict over all tickers.

    data_dict[ticker] must be a DataFrame with HAR features, log_Y, Y_fwd, and any
    columns required by the models (e.g. Returns for GARCH).
    model_dict maps a display name to a model instance with .features, .fit(X, y), .predict(X).
    sample_tickers: if given, fold-by-fold predictions are stored in pred_store for each listed ticker.

    Returns
    -------
    metrics_df  : long-form DataFrame with columns [Ticker, Model, Fold, R2, RMSE, MAE, R2_log, RMSE_log, MAE_log]
    pred_store  : {ticker: DataFrame} with aligned predictions for every ticker in sample_tickers

    Example usage
    -------------
    model_dict = {
        "HAR":          HARLogRegressor(ridge_alpha=0.0),
        "ARIMA(1,0,1)": ARIMALogY(order=(1, 0, 1)),
        "GARCH(1,1)":   GARCHWeeklyRV(p=1, q=1, horizon=5),
    }
    metrics_df, pred_store = run_benchmarks_multi_fold(
        data_dict, model_dict, tickers,
        sample_tickers=["AAPL", "TSLA"],
    )
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
                # model_entry may be a plain model or a (model, remove_outliers_bool) tuple
                if isinstance(model_entry, tuple):
                    model_template, do_remove_outliers = model_entry
                else:
                    model_template, do_remove_outliers = model_entry, False

                # try:
                model  = copy.deepcopy(model_template)
                X_train_fit = remove_outliers(X_train) if do_remove_outliers else X_train
                y_pred = fit_predict_model(model, X_train_fit, X_test)
                m = eval_regression(y_true.loc[test_idx].values, y_pred)
                rows.append({"Ticker": t, "Model": model_name, "Fold": fold_id, **m})

                if t in sample_set:
                    pred_df.loc[test_idx, model_name] = y_pred
                # except Exception:
                #     pass

        if t in sample_set:
            pred_store[t] = pred_df

    metrics_df = pd.DataFrame(rows)
    return metrics_df, pred_store


def summarize_benchmarks(metrics_df: pd.DataFrame) -> pd.DataFrame:
    per_ticker = (
        metrics_df.groupby(["Ticker", "Model"])
        .agg(
            R2=("R2", "mean"), RMSE=("RMSE", "mean"), MAE=("MAE", "mean"),
            R2_log=("R2_log", "mean"), RMSE_log=("RMSE_log", "mean"), MAE_log=("MAE_log", "mean"),
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
            n_tickers=("Ticker", "nunique"),
        )
        .sort_values("median_R2_log", ascending=False)
    )
    return summary


def main():
    import sys
    from pathlib import Path

    # Make sure sibling packages are importable when running directly
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data import get_data_for_har
    from visualize.plot_model_results import plot_ticker_predictions, plot_summary_metrics

    SAMPLE_TICKERS = ["AAPL", "TSLA", "GOOG", "META", "MSFT"]
    RESULTS_DIR = Path(__file__).parent.parent / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading and preprocessing data...")
    data_dict = get_data_for_har(100)
    tickers = list(data_dict.keys())

    model_dict = {
        "HAR":                    (HARLogRegressor(ridge_alpha=0.0), False),
        "HAR (no outliers)":      (HARLogRegressor(ridge_alpha=0.0), True),
        "ARIMA(1,0,1)":           (ARIMALogY(order=(1, 0, 1)),       False),
        "ARIMA(1,0,1) (no outliers)": (ARIMALogY(order=(1, 0, 1)), True),
        "GARCH(1,1)":             (GARCHWeeklyRV(p=1, q=1, scale=1.0, horizon=5), False),
        "GARCH(1,1) (no outliers)": (GARCHWeeklyRV(p=1, q=1, scale=1.0, horizon=5), True),
    }

    print(f"Running cross-validation on {len(tickers)} tickers...")
    metrics_df, pred_store = run_benchmarks_multi_fold(
        data_dict,
        model_dict,
        tickers,
        sample_tickers=SAMPLE_TICKERS,
    )

    # Save raw fold-level metrics
    metrics_df.to_json(RESULTS_DIR / "results_bench.json", orient="records", indent=2)
    print(f"Results saved to {RESULTS_DIR / 'results_bench.json'}")

    # Summarize
    summary = summarize_benchmarks(metrics_df)
    print("\nSummary:\n", summary)

    # Plot 1: summary metrics
    plot_summary_metrics(summary, save_path=str(RESULTS_DIR / "summary_metrics.png"))

    # Plot 2: per-ticker true vs forecast
    plot_ticker_predictions(pred_store, SAMPLE_TICKERS, save_dir=str(RESULTS_DIR))


if __name__ == "__main__":
    main()
