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
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Creates:
      - Y_fwd          : sum of squared returns over the next *horizon* days
                         (excludes today, starts tomorrow).  This is the
                         regression target.
      - RV1            : r_t^2  -- daily realised variance
      - RV5            : past 5-day rolling mean of r^2  (weekly component)
      - RV10           : past 10-day rolling mean of r^2 (bi-weekly component)
      - RV22           : past 22-day rolling mean of r^2 (monthly component)
      - neg_semi_var5  : 5-day rolling mean of r^2 where r < 0
                         (negative semivariance -- captures leverage effect)
      - pos_semi_var5  : 5-day rolling mean of r^2 where r >= 0
                         (positive semivariance)
      - log_*          : natural log of each RV measure + eps for stability.
                         log_Y is the preferred training target for HAR models.

    Feature alignment:
      All RV features use *strictly past* data (rolling windows end at t).
      Y_fwd uses *strictly future* data (r_{t+1}^2 ... r_{t+horizon}^2).
      => No look-ahead leakage between features and target.

    Scaling:
      No StandardScaler is applied here.  Fitting a global scaler before
      cross-validation splits would leak test-period statistics into training
      features.  Models that need standardised inputs should wrap themselves
      in a sklearn Pipeline([StandardScaler, regressor]) so the scaler is
      re-fit on each fold's training data only.

    Index safety:
      When called via preprocess_data(), ensure_datetime() has already been
      applied and the index is a DatetimeIndex named "Date".  Calling
      ensure_datetime() again would raise a KeyError because "Date" is no
      longer a column.  We therefore only call ensure_datetime() when the
      index is not already a DatetimeIndex -- safe to call standalone too.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = ensure_datetime(df)
    else:
        df = df.copy().sort_index()
    out = df.copy()
    r = pd.to_numeric(out[returns_col], errors="coerce")

    r2 = r ** 2

    # ── Past RV components (HAR hierarchy) ────────────────────────────────
    out["RV1"]  = r2
    out["RV5"]  = r2.rolling(5).mean()
    out["RV10"] = r2.rolling(10).mean()
    out["RV22"] = r2.rolling(22).mean()

    # ── Asymmetric / leverage components ──────────────────────────────────
    # Negative semivariance: squared returns on down-days only.
    # Captures the leverage effect (bad-news days drive future vol more than
    # good-news days of equal magnitude).
    neg_r2 = r2.where(r < 0, other=0.0)
    pos_r2 = r2.where(r >= 0, other=0.0)
    out["neg_semi_var5"] = neg_r2.rolling(5).mean()
    out["pos_semi_var5"] = pos_r2.rolling(5).mean()

    # ── Forward target ────────────────────────────────────────────────────
    # Y_fwd_t = r_{t+1}^2 + r_{t+2}^2 + ... + r_{t+horizon}^2
    # shift(-k) pulls future row k into the current index position, so the
    # value stored at date t is a pure future quantity.
    y = sum((r.shift(-k) ** 2) for k in range(1, horizon + 1))
    out["Y_fwd"] = y

    # ── Log transforms (single eps for consistency) ───────────────────────
    out["log_RV1"]       = np.log(out["RV1"]          + eps)
    out["log_RV5"]       = np.log(out["RV5"]          + eps)
    out["log_RV10"]      = np.log(out["RV10"]         + eps)
    out["log_RV22"]      = np.log(out["RV22"]         + eps)
    out["log_neg_semi5"] = np.log(out["neg_semi_var5"] + eps)
    out["log_pos_semi5"] = np.log(out["pos_semi_var5"] + eps)
    out["log_Y"]         = np.log(out["Y_fwd"]        + eps)

    # Drop rows that are missing any required feature or the target
    required = [
        "Y_fwd", "RV1", "RV5", "RV10", "RV22",
        "neg_semi_var5", "pos_semi_var5",
    ]
    out = out.dropna(subset=required)
    return out


def preprocess_data(df: pd.DataFrame, lag_list: Optional[List[int]] = None):
    df = ensure_datetime(df)
    df = add_returns_column(df)
    if lag_list is not None:
        df = add_lag_features(df, lag_list)
    df = add_forward_rv_target_and_past_rv_features(df)
    return df


def preprocess(data: dict, lag_list: Optional[List[int]] = None, preprocess_func=None):
    if preprocess_func is None:
        preprocess_func = preprocess_data
    preprocessed_data = {}
    for ticker, df in data.items():
        preprocessed_data[ticker] = preprocess_func(df, lag_list)
    return preprocessed_data


def preprocess_df_for_har(df: pd.DataFrame, lag_list: Optional[List[int]] = None):
    """
    Preprocess a single ticker DataFrame for HAR-family models.

    No StandardScaler is applied here; see add_forward_rv_target_and_past_rv_features
    for the full explanation.  Scaling is handled inside each model Pipeline.
    """
    df = preprocess_data(df, lag_list=lag_list)
    if df.empty or len(df) < 2:
        raise Exception("DataFrame is empty or too small after preprocessing.")
    return df


def preprocess_for_har(data: dict, lag_list: Optional[List[int]] = None):
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
                ["log_RV1", "log_RV5", "log_RV22", "log_Y"].  Only columns
                that actually exist in df are checked.
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