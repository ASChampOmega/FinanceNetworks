"""
data/preprocess_index.py
========================
Preprocessing utilities for the Oxford-Man Realized Volatility Indices dataset.

Unlike the stock pipeline (preprocess.py) which must *construct* daily realized
variance from squared returns, this dataset supplies high-quality 5-minute
realized-variance (RV) estimates directly.  We therefore:

  - Use the .rv column as RV1 (daily realized variance).
  - Compute RV5/10/22 as rolling means of RV1 (not of squared returns).
  - Compute Y_fwd as the sum of the next *horizon* days' RV1.
  - Use SPX2 as the market proxy for market-level features.

The feature set is aligned with the existing pipeline so that the same HAR /
network models can be used without modification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from data.preprocess import ensure_datetime, remove_outliers  # noqa: F401 – re-exported

# ── Constants ─────────────────────────────────────────────────────────────────

TICKERS = [
    "SPX2", "FTSE2", "N2252", "GDAXI2", "RUT2", "AORD2", "DJI2", "IXIC2",
    "FCHI2", "HSI2", "KS11", "AEX", "SSMI", "IBEX2", "NSEI", "MXX", "BVSP",
    "GSPTSE", "STOXX50E", "FTSTI", "FTSEMIB",
]

_CSV_FILE = "OxfordManRealizedVolatilityIndices.csv"
MARKET_TICKER = "SPX2"


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_oxford_man_data(
    path: "str | Path | None" = None,
) -> pd.DataFrame:
    """
    Load the Oxford-Man CSV and return a wide DataFrame indexed by Date.

    The CSV has a 3-row header.  Row 2 (0-indexed) contains short column codes
    like ``SPX2.rv``, ``SPX2.r``, etc.  We use ``header=2`` to grab those
    directly as column names.
    """
    if path is None:
        path = Path(__file__).parent / "data_files" / _CSV_FILE
    path = Path(path)

    raw = pd.read_csv(path, header=2)

    # First column is "DateID" (YYYYMMDD integer dates)
    first_col = raw.columns[0]
    raw = raw.rename(columns={first_col: "DateID"})
    raw["Date"] = pd.to_datetime(raw["DateID"].astype(str), format="%Y%m%d")
    raw = raw.set_index("Date").sort_index()
    raw = raw.drop(columns=["DateID"], errors="ignore")
    return raw


def extract_index_data(
    raw_df: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Pull the columns for *ticker* out of the wide raw DataFrame.

    Returns a DataFrame with columns: rv, r, closeprice  (renamed from
    ``<ticker>.rv``, ``<ticker>.r``, ``<ticker>.closeprice``).
    """
    cols = {
        f"{ticker}.rv": "rv",
        f"{ticker}.r": "r",
        f"{ticker}.closeprice": "closeprice",
        f"{ticker}.rs": "rs",           # realized semivariance (downside)
    }
    existing = {k: v for k, v in cols.items() if k in raw_df.columns}
    if not existing:
        raise KeyError(f"No columns found for ticker '{ticker}'")

    sub = raw_df[list(existing.keys())].rename(columns=existing).copy()
    # Coerce to numeric (some cells may be empty / non-numeric)
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub


# ── Feature engineering ───────────────────────────────────────────────────────

def preprocess_index_data(
    df: pd.DataFrame,
    horizon: int = 5,
    eps: float = 1e-8,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Build HAR features from actual 5-minute realized-variance data.

    Expected input columns: ``rv`` (realized variance), ``r`` (return),
    and optionally ``rs`` (realized semivariance, downside).

    Creates the same column names as the stock pipeline so that existing
    models work out of the box:

      RV1, RV5, RV10, RV22          – past RV components
      neg_semi_var5, pos_semi_var5   – asymmetric / leverage components
      Y_fwd, log_Y                  – forward target
      Returns                       – for network feature_cols compatibility
      log_RV1 ... log_pos_semi5     – log transforms
    """
    out = df.copy()

    rv = pd.to_numeric(out["rv"], errors="coerce")
    r = pd.to_numeric(out["r"], errors="coerce")

    # ── Returns (kept as-is; decimal returns × 100 for parity with the
    #    stock pipeline which uses 100 * log-return) ──────────────────────
    out["Returns"] = r * 100.0

    # ── Past RV components (HAR hierarchy) ────────────────────────────────
    out["RV1"]  = rv
    out["RV5"]  = rv.rolling(5).mean()
    out["RV10"] = rv.rolling(10).mean()
    out["RV22"] = rv.rolling(22).mean()

    # ── Asymmetric / leverage components ─────────────────────────────────
    # Use actual RV weighted by sign of the contemporaneous return.
    # neg_semi_var5: average RV on down-days (r < 0) captures leverage effect.
    neg_rv = rv.where(r < 0, other=0.0)
    pos_rv = rv.where(r >= 0, other=0.0)
    out["neg_semi_var5"] = neg_rv.rolling(5).mean()
    out["pos_semi_var5"] = pos_rv.rolling(5).mean()

    # ── Forward target ────────────────────────────────────────────────────
    # Y_fwd_t = rv_{t+1} + rv_{t+2} + ... + rv_{t+horizon}
    y = sum(rv.shift(-k) for k in range(1, horizon + 1))
    out["Y_fwd"] = y

    # ── Log transforms ────────────────────────────────────────────────────
    out["log_RV1"]       = np.log(out["RV1"]          + eps)
    out["log_RV5"]       = np.log(out["RV5"]          + eps)
    out["log_RV10"]      = np.log(out["RV10"]         + eps)
    out["log_RV22"]      = np.log(out["RV22"]         + eps)
    out["log_neg_semi5"] = np.log(out["neg_semi_var5"] + eps)
    out["log_pos_semi5"] = np.log(out["pos_semi_var5"] + eps)
    out["log_Y"]         = np.log(out["Y_fwd"]        + eps)

    required = [
        "Y_fwd", "RV1", "RV5", "RV10", "RV22",
        "neg_semi_var5", "pos_semi_var5",
    ]
    if drop_na:
        out = out.dropna(subset=required)
    return out


def add_index_market_features(
    df: pd.DataFrame,
    market_rv: pd.Series,
    market_returns: pd.Series,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Add market-level features using actual SPX2 realized-variance data.

    Creates the same columns as ``add_market_features`` in preprocess.py
    so that existing models work without modification.
    """
    df = df.copy()
    df["Market_Returns"] = market_returns.reindex(df.index)
    mrv = market_rv.reindex(df.index)
    df["Market_RV5"]      = mrv.rolling(5).mean()
    df["Market_RV22"]     = mrv.rolling(22).mean()
    df["log_Market_RV5"]  = np.log(df["Market_RV5"]  + eps)
    df["log_Market_RV22"] = np.log(df["Market_RV22"] + eps)
    return df


# ── Main entry point ─────────────────────────────────────────────────────────

def get_index_data_for_har(
    csv_path: "str | Path | None" = None,
    tickers: "list[str] | None" = None,
) -> dict[str, pd.DataFrame]:
    """
    End-to-end preprocessing: load CSV → per-index features → market features.

    Returns ``{ticker: DataFrame}`` ready for HAR / network models.
    """
    raw = load_oxford_man_data(csv_path)
    if tickers is None:
        tickers = TICKERS

    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            sub = extract_index_data(raw, ticker)
        except KeyError:
            print(f"[preprocess_index] Skipping {ticker}: columns not found.")
            continue

        # Forward-fill first (standard for financial panel data), then drop
        # rows where rv or r is still NaN (likely start / end of series)
        sub = sub.ffill()
        sub = sub.dropna(subset=["rv", "r"])
        if len(sub) < 300:
            print(f"[preprocess_index] Skipping {ticker}: only {len(sub)} rows.")
            continue

        processed = preprocess_index_data(sub, drop_na=False)

        # Final dropna: require all model-critical columns present
        required = [
            "Y_fwd", "RV1", "RV5", "RV10", "RV22",
            "neg_semi_var5", "pos_semi_var5",
        ]
        processed = processed.dropna(subset=required)
        if processed.empty or len(processed) < 300:
            print(f"[preprocess_index] Skipping {ticker}: too few rows after preprocessing.")
            continue

        result[ticker] = processed

    return result
