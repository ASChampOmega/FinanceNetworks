import pandas as pd
import numpy as np
from typing import List, Optional

def add_returns_column(df: pd.DataFrame):
    df["Returns"] = 100 * np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    return df

def add_lag_features(df: pd.DataFrame, lag_list: List[int]):
    for lag in lag_list:
        df[f"Returns_Lag_{lag}"] = 100 * np.log(df["Adj Close"] / df["Adj Close"].shift(lag))
    return df

def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df.set_index("Date")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    return df

def add_forward_rv_target_and_past_rv_features(
    df: pd.DataFrame,
    returns_col: str = "Returns",
    horizon: int = 5,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Creates:
      - Y_fwd: sum of squared returns over next horizon days (exclude today, start tomorrow)
      - RV1: r_t^2
      - RV5: mean of last 5 squared returns (past)
      - RV22: mean of last 22 squared returns (past)
      - logs of all (safe with eps)
    """
    df = ensure_datetime(df)
    out = df.copy()
    r = pd.to_numeric(out[returns_col], errors="coerce")

    r2 = r ** 2
    out["RV1"] = r2
    out["RV5"] = r2.rolling(5).mean()
    out["RV22"] = r2.rolling(22).mean()

    # Future realized variance over next horizon days (exclude today, start tomorrow)
    y = sum((r.shift(-k) ** 2) for k in range(1, horizon + 1))
    out["Y_fwd"] = y

    # Logs (often better behaved)
    out["log_RV1"] = np.log(out["RV1"] + eps)
    out["log_RV5"] = np.log(out["RV5"] + eps)
    out["log_RV22"] = np.log(out["RV22"] + eps)
    out["log_Y"] = np.log(out["Y_fwd"] + eps)

    # Drop rows that can’t be used (need enough past + enough future)
    out = out.dropna(subset=["Y_fwd", "RV1", "RV5", "RV22"])
    return out

def preprocess_data(df: pd.DataFrame, lag_list: Optional[List[int]] = None):
    df = ensure_datetime(df)
    df = add_returns_column(df)
    if lag_list is not None:
        df = add_lag_features(df, lag_list)
    df = add_forward_rv_target_and_past_rv_features(df)
    return df

def preprocess(data: dict[str, pd.DataFrame], lag_list: Optional[List[int]] = None, preprocess_func = None):
    if preprocess_func is None:
        preprocess_func = preprocess_data
    preprocessed_data = {}
    for ticker, df in data.items():
        preprocessed_data[ticker] = preprocess_func(df, lag_list)
    return preprocessed_data

def preprocess_df_for_har(df: pd.DataFrame, lag_list: Optional[List[int]] = None):
    from sklearn.preprocessing import StandardScaler
    df = preprocess_data(df, lag_list=lag_list)
    df['log_Y'] = np.log(df['Y_fwd'] + 1e-8)
    df['log_RV1'] = np.log(df['RV1'] + 1e-8)
    df['log_RV5'] = np.log(df['RV5'] + 1e-8)
    df['log_RV22'] = np.log(df['RV22'] + 1e-8)
    # Standardize HAR input features per-ticker for numerical stability.
    # log_Y and Y_fwd are intentionally left unscaled: log_Y is the training
    # target (models exp() it back), and Y_fwd is the evaluation target.
    # Returns is left unscaled so GARCH can work in its natural percentage-return units.
    har_cols = ['log_RV1', 'log_RV5', 'log_RV22']
    if df.empty or len(df) < 2:
        raise Exception("DataFrame is empty or too small after preprocessing.")
    scaler = StandardScaler()
    df[har_cols] = scaler.fit_transform(df[har_cols])
    return df


def preprocess_for_har(data: dict[str, pd.DataFrame], lag_list: Optional[List[int]] = None):
    result = {}
    for ticker, df in data.items():
        try:
            processed = preprocess_df_for_har(df, lag_list=lag_list)
            if processed.empty:
                print(f"[preprocess_for_har] Skipping {ticker}: empty after preprocessing.")
                continue
            result[ticker] = processed
        except Exception as e:
            print(f"[preprocess_for_har] Skipping {ticker}: {e}")
    return result

def remove_outliers(
    df: pd.DataFrame,
    cols: "str | List[str]" = None,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """
    Remove rows where any of the specified columns has a value more than
    z_thresh standard deviations from its mean.

    Parameters
    ----------
    df        : DataFrame to filter.
    cols      : Column name or list of column names to check.  Defaults to
                ["log_RV1", "log_RV5", "log_RV22", "log_Y"] (the HAR
                feature/target columns).  Only columns that exist in df are
                used.
    z_thresh  : Rows whose z-score magnitude exceeds this value in *any*
                of the checked columns are dropped.
    """
    if df.empty:
        return df
    if cols is None:
        cols = ["log_RV1", "log_RV5", "log_RV22", "log_Y"]
    if isinstance(cols, str):
        cols = [cols]
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        std = df[col].std()
        if std == 0:
            continue
        z_scores = (df[col] - df[col].mean()) / std
        mask &= np.abs(z_scores) < z_thresh
    return df[mask]

