"""
data/volatility_spikes.py
=========================
Preprocessing utilities for volatility-spike classification.

A *volatility spike* at time t is defined as a forward realized-variance
observation (Y_fwd) that lies above the *spike_quantile*-th quantile of the
training-fold Y_fwd distribution.  The threshold is always computed on
training data inside each cross-validation fold so no test-period statistics
leak into the labelling.

Functions
---------
compute_spike_threshold          : Compute quantile threshold from training Y_fwd.
add_spike_label                  : Attach binary 'spike' column given a threshold.
preprocess_for_classification    : Full pipeline for one ticker DataFrame.
preprocess_all_for_classification: Apply across a {ticker: DataFrame} dict.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, List

from data.preprocess import preprocess_data


# ---------------------------------------------------------------------------
# Threshold and labelling helpers
# ---------------------------------------------------------------------------

def compute_spike_threshold(
    y_train: pd.Series,
    quantile: float = 0.8,
) -> float:
    """
    Return the *quantile*-th quantile of *y_train* (Y_fwd values in
    variance scale).

    Parameters
    ----------
    y_train  : Series of forward-RV values from the training fold.
    quantile : Fraction at or above which an observation is a spike.
               0.8 → top-20% of the training distribution is a spike.

    Returns
    -------
    threshold : float scalar used to label spikes in both train and test sets.
    """
    return float(np.nanquantile(y_train.values, quantile))


def add_spike_label(
    df: pd.DataFrame,
    threshold: float,
    target_col: str = "Y_fwd",
    label_col: str = "spike",
) -> pd.DataFrame:
    """
    Add a binary *label_col* column: 1 if *target_col* > *threshold*, 0 otherwise.

    Parameters
    ----------
    df         : DataFrame containing *target_col*.
    threshold  : Spike threshold (scalar, typically from compute_spike_threshold).
    target_col : Column to compare against the threshold (default 'Y_fwd').
    label_col  : Name of the new binary column (default 'spike').

    Returns
    -------
    Copy of *df* with the new binary column appended.
    """
    df = df.copy()
    df[label_col] = (df[target_col] > threshold).astype(int)
    return df


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_for_classification(
    df: pd.DataFrame,
    quantile: float = 0.8,
    lag_list: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for volatility-spike classification.

    Steps
    -----
    1. Call ``preprocess_data()`` from preprocess.py to compute returns,
       HAR features (RV1 / RV5 / RV10 / RV22 / semivariances), log
       transforms, and the forward-RV target Y_fwd.
    2. Attach a global spike label based on the full-history Y_fwd
       distribution of this ticker.  This label is used for inspection and
       stratified train/test splitting.

    Note on leakage
    ---------------
    The *global* spike column added here is NOT used directly as the cross-
    validation target.  The CV loop in classification.py recomputes the
    threshold from training-fold Y_fwd values on every fold so that test-
    period statistics never influence the labelling.

    Parameters
    ----------
    df       : Raw OHLCV DataFrame for one ticker.
    quantile : Upper-tail quantile defining a spike (default 0.8 → top-20 %).
    lag_list : Optional lag list forwarded to preprocess_data.

    Returns
    -------
    Preprocessed DataFrame with HAR features, Y_fwd, log transforms, and
    a 'spike' column based on the global per-ticker threshold.
    """
    df = preprocess_data(df, lag_list=lag_list)
    if df.empty:
        return df

    threshold = compute_spike_threshold(df["Y_fwd"], quantile=quantile)
    df = add_spike_label(df, threshold=threshold, label_col="spike")
    return df


def preprocess_all_for_classification(
    data: dict,
    quantile: float = 0.8,
    lag_list: Optional[List[int]] = None,
) -> dict:
    """
    Apply ``preprocess_for_classification`` to every ticker in *data*.

    Tickers whose DataFrames are empty or raise an exception are silently
    skipped (a warning is printed).

    Parameters
    ----------
    data     : {ticker: raw_df} mapping.
    quantile : Spike quantile (forwarded to preprocess_for_classification).
    lag_list : Optional lag list (forwarded).

    Returns
    -------
    {ticker: processed_df} mapping with only successfully processed tickers.
    """
    result: dict = {}
    for ticker, df in data.items():
        try:
            processed = preprocess_for_classification(
                df, quantile=quantile, lag_list=lag_list
            )
            if processed.empty or len(processed) < 2:
                print(
                    f"[preprocess_for_classification] Skipping {ticker}: "
                    "empty after preprocessing."
                )
                continue
            result[ticker] = processed
        except Exception as exc:
            print(f"[preprocess_for_classification] Skipping {ticker}: {exc}")
    return result
