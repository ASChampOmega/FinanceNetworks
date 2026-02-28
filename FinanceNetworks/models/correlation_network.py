"""
correlation_network.py
======================
Financial correlation-network feature extractors for volatility forecasting.

Architecture
------------
FinanceNetworkBase  (abstract sklearn Estimator + Transformer)
    |
    +-- _compute_distance_matrix()   ABSTRACT: (n x n) distance DataFrame
    +-- _build_graph()               dispatches to threshold / kNN builders
    +-- fit(data_dict)               builds ALL rolling graph snapshots ONCE
    +-- transform(data_dict)         appends net_* feature columns to every ticker
    +-- slice_features(result, idx)  slices transform() output to a date index

Concrete subclasses
    SquaredCorrelationNetwork      distance = 1 - rho^2
    PartialCorrelationNetwork      distance = 1 - pcorr^2  (via precision matrix)

Graph types  (graph_type parameter)
    "threshold"  -- edge (i,j) iff d_ij < threshold
    "knn"        -- k-Nearest Neighbours (undirected, symmetrised)

Key parameter: window
    At each rebuild date t the graph is estimated from the T most recent
    returns: r_{t-window+1}, ..., r_t.  All data is strictly in the past,
    so there is no look-ahead leakage.

Features appended to each ticker DataFrame (all prefixed net_)
    net_degree              node degree in the current snapshot
    net_degree_change       neighbourhood turnover at snapshot t:
                              1 - |N_{i,t-1} ∩ N_{i,t}| / max(1, |N_{i,t}|)
                            0 = identical neighbourhood (stable),
                            1 = complete rewiring (no shared neighbours).
                            NaN at the first snapshot.
    net_idw_<col>           IDW mean of <col> over graph neighbours:
                              sum_j (1/d_ij)*col_j(t) / sum_j (1/d_ij)
                            one column per entry in feature_cols

Leakage safety
--------------
Graph at t  : built from r_{t-window+1}...r_t          -- strictly past
IDW at t    : reads log_RV of neighbours at date t      -- past RV only
Target      : Y_fwd_t = sum r_{t+k}^2, k=1..horizon    -- strictly future
Between rebuild dates, features are forward-filled (last-observation-carried-
forward), so every row has a value.  Rows before the first snapshot are NaN.

Recommended usage (offline, once per experiment)
-------------------------------------------------
    # 1. Build ALL snapshots ONCE on the full dataset -- O(T/step) graphs.
    #    Because every snapshot uses only past returns, fitting on the full
    #    history does not leak future information into the graph topology.
    net = SquaredCorrelationNetwork(window=60, step=5, graph_type="knn", k=5)
    full_features = net.fit_transform(full_data_dict)

    # 2. Inside the CV loop, slice by date -- O(1), no re-fitting needed.
    for train_idx, test_idx in folds:
        train_dict = FinanceNetworkBase.slice_features(full_features, train_idx)
        test_dict  = FinanceNetworkBase.slice_features(full_features, test_idx)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Distance-matrix helpers
# ---------------------------------------------------------------------------

def _safe_corr(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Pearson correlation matrix of the columns of *arr*.
    Clips to [-1+eps, 1-eps] so downstream distance formulae stay real.
    Falls back to the identity matrix if the result is non-finite.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = np.corrcoef(arr, rowvar=False)
    if not np.isfinite(c).all():
        c = np.eye(arr.shape[1])
    return np.clip(c, -1.0 + eps, 1.0 - eps)


def _squared_corr_distance(arr: np.ndarray) -> np.ndarray:
    """
    d_ij = 1 - rho_ij^2

    Properties
    ----------
    Range [0, 1].  d=0 when |rho|=1 (perfect co-/anti-movement),
    d=1 when rho=0 (uncorrelated).  Treats positive and negative correlation
    symmetrically, which is desirable: a strongly anti-correlated pair is just
    as structurally "close" in the network as a positively correlated one.
    Equivalent to 1 - R^2 (fraction of variance not explained by the other).
    """
    corr = _safe_corr(arr)
    return 1.0 - corr ** 2


def _partial_corr_distance(arr: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """
    d_ij = 1 - pcorr_ij^2

    Partial correlations come from the precision matrix Omega = Sigma^{-1}:
        pcorr_ij = -Omega_ij / sqrt(Omega_ii * Omega_jj)

    Conditioning out all other stocks gives a measure of *direct* dependency.
    The correlation matrix is regularised with diagonal shrinkage before
    inversion; shrinkage is increased automatically when n_obs <= n_vars.
    """
    corr = _safe_corr(arr)
    n_obs, n_vars = arr.shape

    alpha = shrinkage if n_obs > n_vars else min(shrinkage + 0.2, 0.5)
    corr_reg = (1.0 - alpha) * corr + alpha * np.eye(n_vars)

    try:
        precision = np.linalg.inv(corr_reg)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(corr_reg)

    d_diag = np.sqrt(np.abs(np.diag(precision)).clip(1e-10))
    pcorr = -precision / np.outer(d_diag, d_diag)
    np.fill_diagonal(pcorr, 1.0)
    pcorr = np.clip(pcorr, -1.0 + 1e-8, 1.0 - 1e-8)

    return 1.0 - pcorr ** 2


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _build_threshold_graph(dist_df: pd.DataFrame, threshold: float) -> nx.Graph:
    """
    Add edge (i, j) iff d_ij < threshold.

    A lower threshold gives a sparser graph (only the most similar pairs are
    connected).  The self-distance diagonal is ignored.  If no pairs satisfy
    the threshold the graph will have no edges (all isolates).
    """
    tickers = dist_df.columns.tolist()
    arr = dist_df.values
    n = len(tickers)
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(arr[i, j])
            if np.isfinite(w) and w < threshold:
                G.add_edge(tickers[i], tickers[j], weight=w)
    return G


def _build_knn_graph(dist_df: pd.DataFrame, k: int) -> nx.Graph:
    """
    k-Nearest Neighbours graph (undirected union of the directed kNN digraph).

    Each node is connected to its k nearest neighbours.  Symmetrisation means
    some nodes may exceed degree k.  Self-loops are excluded.
    """
    tickers = dist_df.columns.tolist()
    arr = dist_df.values.copy()
    np.fill_diagonal(arr, np.inf)
    n = len(tickers)
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for i in range(n):
        for j in np.argsort(arr[i])[:k]:
            w = float(arr[i, j])
            if np.isfinite(w):
                G.add_edge(tickers[i], tickers[j], weight=w)
    return G


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class FinanceNetworkBase(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base for rolling financial correlation networks.

    Parameters
    ----------
    window : int
        Number of trading days used to estimate the distance matrix at each
        rebuild.  At snapshot date t the window covers r_{t-window+1}...r_t.
        Rule of thumb: window >= 2 * n_tickers for the correlation matrix to
        be numerically full-rank.  Default: 60 (~3 months).
    step : int
        Rebuild the graph every *step* trading days.  step=1 is maximally
        fresh but expensive; step=5 (weekly) is a good default.
    graph_type : {"threshold", "knn"}
        Sparsification method applied to the full distance matrix.
    threshold : float
        Edge cutoff for graph_type="threshold". Edges with d_ij >= threshold
        are dropped.  Typical range for 1-rho^2: 0.1 (dense) -- 0.5 (sparse).
    k : int
        Number of nearest neighbours for graph_type="knn".
    feature_cols : list of str
        Columns from each ticker DataFrame to aggregate via IDW over neighbours.
        Defaults to ["log_RV1", "log_RV5", "log_RV22"].
    returns_col : str
        Column containing log-returns used to build the distance matrix.
    min_obs_frac : float
        Minimum fraction of non-NaN observations a ticker must have in a
        window to be included in that snapshot's graph.
    min_tickers : int
        Skip a snapshot entirely if fewer tickers pass the min_obs_frac filter.
    """

    _GRAPH_BUILDERS = {"threshold", "knn"}

    def __init__(
        self,
        window: int = 60,
        step: int = 5,
        graph_type: str = "knn",
        threshold: float = 0.3,
        k: int = 5,
        feature_cols: Optional[List[str]] = None,
        returns_col: str = "Returns",
        min_obs_frac: float = 0.8,
        min_tickers: int = 10,
    ):
        if graph_type not in self._GRAPH_BUILDERS:
            raise ValueError(
                f"graph_type must be one of {sorted(self._GRAPH_BUILDERS)}; "
                f"got '{graph_type}'"
            )
        self.window       = window
        self.step         = step
        self.graph_type   = graph_type
        self.threshold    = threshold
        self.k            = k
        self.feature_cols = feature_cols or ["log_RV1", "log_RV5", "log_RV22"]
        self.returns_col  = returns_col
        self.min_obs_frac = min_obs_frac
        self.min_tickers  = min_tickers

        # Populated by fit()
        self.tickers_: List[str] = []
        # Ordered list of (rebuild_date, nx.Graph).  The graph at index i covers
        # returns ending on rebuild_date[i]; features are forward-filled to the
        # next rebuild date.
        self._snapshots: List[Tuple[pd.Timestamp, nx.Graph]] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_distance_matrix(self, returns_window: pd.DataFrame) -> pd.DataFrame:
        """
        Subclasses implement their distance/dissimilarity measure here.

        Parameters
        ----------
        returns_window : DataFrame, shape (window, n_tickers)
            Log-returns for the tickers that passed the min_obs_frac filter,
            NaNs already filled with 0.0.

        Returns
        -------
        dist_df : symmetric DataFrame, shape (n_tickers, n_tickers)
            Pairwise distances in [0, 1].  Diagonal entries are 0.
        """

    # ------------------------------------------------------------------
    # Graph builder dispatch
    # ------------------------------------------------------------------

    def _build_graph(self, dist_df: pd.DataFrame) -> nx.Graph:
        if self.graph_type == "threshold":
            return _build_threshold_graph(dist_df, self.threshold)
        return _build_knn_graph(dist_df, self.k)

    # ------------------------------------------------------------------
    # Fit: build ALL rolling snapshots once
    # ------------------------------------------------------------------

    def fit(
        self,
        data_dict: Dict[str, pd.DataFrame],
        y=None,
    ) -> "FinanceNetworkBase":
        """
        Build the full sequence of rolling graph snapshots from *data_dict*.

        Intended to be called ONCE on the complete dataset (training + test
        combined) before cross-validation begins.  Each snapshot at date t
        only uses returns up to and including t, so calling fit() on the full
        history does NOT leak future returns into the graph topology.

        Steps
        -----
        1. Coalesce per-ticker DataFrames into a single wide (dates x tickers)
           returns matrix using only *returns_col*. This keeps memory minimal --
           the feature columns (log_RV*, etc.) are NOT loaded here.
        2. Identify rebuild dates: every *step*-th date after the first full
           window of size *window*.
        3. At each rebuild date t:
           a. Slice the trailing *window* rows of the wide returns matrix.
           b. Drop tickers with too many NaNs, fill remaining NaNs with 0.
           c. Call _compute_distance_matrix() -> dist_df.
           d. Call _build_graph(dist_df) -> G.
           e. Append (t, G) to self._snapshots.

        After fit(), call transform() once to materialise all features, and
        then use slice_features() to split by CV fold -- no re-fitting needed.
        """
        # ── 1. Coalesce returns into a wide matrix ─────────────────────
        series: Dict[str, pd.Series] = {}
        for ticker, df in data_dict.items():
            if self.returns_col not in df.columns:
                continue
            idx = (df.index if isinstance(df.index, pd.DatetimeIndex)
                   else pd.to_datetime(df.index))
            series[ticker] = pd.Series(
                df[self.returns_col].values, index=idx, name=ticker
            )

        if len(series) < self.min_tickers:
            raise ValueError(
                f"Only {len(series)} tickers have '{self.returns_col}'; "
                f"need >= min_tickers={self.min_tickers}."
            )

        returns_wide = pd.DataFrame(series).sort_index()
        self.tickers_ = returns_wide.columns.tolist()

        all_dates = returns_wide.index
        if len(all_dates) < self.window:
            raise ValueError(
                f"Only {len(all_dates)} dates available; "
                f"need >= window={self.window}."
            )

        # ── 2. Rolling graph construction ──────────────────────────────
        self._snapshots = []
        rebuild_dates = all_dates[self.window - 1 :: self.step]

        for date in tqdm(
            rebuild_dates,
            desc=f"Fitting {type(self).__name__} [window={self.window}, {self.graph_type}]",
            unit="snap",
        ):
            loc = all_dates.get_loc(date)
            window_ret = returns_wide.iloc[loc - self.window + 1 : loc + 1]

            ok = window_ret.columns[
                window_ret.notna().mean() >= self.min_obs_frac
            ]
            if len(ok) < self.min_tickers:
                continue

            window_ret = window_ret[ok].fillna(0.0)

            try:
                dist_df = self._compute_distance_matrix(window_ret)
                G = self._build_graph(dist_df)
                self._snapshots.append((date, G))
            except Exception as exc:
                warnings.warn(f"Snapshot at {date.date()} skipped: {exc}")

        if not self._snapshots:
            raise RuntimeError(
                "No valid graph snapshots were built.  "
                "Check window size, min_tickers, and data coverage."
            )

        return self

    # ------------------------------------------------------------------
    # Transform: feature extraction
    # ------------------------------------------------------------------

    def transform(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Materialise network features for every ticker in *data_dict*.

        Call this ONCE on the full dataset after fit(), then use
        slice_features() to split by date inside the CV loop.

        Features appended
        -----------------
        net_degree : float
            Node degree in the current snapshot.

        net_degree_change : float
            Fractional degree change vs the previous snapshot:
              (deg_t - deg_{t-1}) / max(deg_{t-1}, 1)
            NaN at the first snapshot.  A large positive spike means the stock
            suddenly gained many connections -- a potential contagion signal.

        net_idw_<col> : float   (one per entry in feature_cols)
            Inverse-distance-weighted mean of <col> across graph neighbours j:
              sum_j w_j * col_j(t)  /  sum_j w_j,   w_j = 1 / d_ij
            Uses the neighbour's own-date value (past RV, not future).

        Performance note
        ----------------
        Before iterating over snapshots, the feature columns are pre-aligned
        to ALL snapshot dates in a single vectorised reindex + ffill operation.
        This avoids O(T) linear scans inside the inner neighbour loop, giving
        O(1) lookup per (snapshot, ticker, neighbour, feature_col).
        """
        if not self._snapshots:
            raise RuntimeError("Call fit() before transform().")

        snap_dates = pd.DatetimeIndex([d for d, _ in self._snapshots])

        # ── Pre-compute IDW lookup tables ──────────────────────────────
        # For each feature_col: wide (dates x tickers) --> ffill --> reindex
        # to snapshot dates so lookup is simply wf_snaps.loc[date, ticker].
        wide_at_snaps: Dict[str, pd.DataFrame] = {}
        for col in self.feature_cols:
            sdict = {}
            for ticker, df in data_dict.items():
                if col not in df.columns:
                    continue
                idx = (df.index if isinstance(df.index, pd.DatetimeIndex)
                       else pd.to_datetime(df.index))
                sdict[ticker] = pd.Series(df[col].values, index=idx, name=ticker)
            if not sdict:
                continue
            # Build full wide matrix, forward-fill, then snap to rebuild dates
            wide = pd.DataFrame(sdict).sort_index()
            # Union of all ticker dates plus snap dates so ffill can propagate
            combined = wide.reindex(
                wide.index.union(snap_dates).sort_values()
            ).ffill()
            wide_at_snaps[col] = combined.reindex(snap_dates)

        # ── Compute per-snapshot per-ticker feature records ────────────
        all_net_cols = (
            ["net_degree", "net_degree_change"]
            + [f"net_idw_{c}" for c in self.feature_cols]
        )

        snap_records: Dict[pd.Timestamp, Dict[str, Dict[str, float]]] = {}
        prev_neighbors: Dict[str, set] = {}   # stores N_{i,t-1} for each ticker

        for date, G in self._snapshots:
            recs: Dict[str, Dict[str, float]] = {}

            for ticker in data_dict:
                if ticker not in G.nodes:
                    continue

                deg = int(G.degree(ticker))
                current_nb = set(G.neighbors(ticker))
                prev_nb = prev_neighbors.get(ticker)

                # Neighborhood turnover (Jaccard-recall style):
                #   1 - |N_{t-1} ∩ N_t| / max(1, |N_t|)
                # 0 = identical neighbourhood (stable),
                # 1 = complete rewiring (no shared neighbours).
                # NaN at the very first snapshot of each ticker.
                deg_change = (
                    1.0 - len(prev_nb & current_nb) / max(1, len(current_nb))
                    if prev_nb is not None
                    else np.nan
                )

                rec: Dict[str, float] = {
                    "net_degree":        float(deg),
                    "net_degree_change": deg_change,
                }

                neighbours = list(G.neighbors(ticker))
                for col, wf in wide_at_snaps.items():
                    feat_key = f"net_idw_{col}"
                    if not neighbours:
                        rec[feat_key] = np.nan
                        continue

                    vals, weights = [], []
                    for nb in neighbours:
                        if nb not in wf.columns:
                            continue
                        val = wf.loc[date, nb]   # O(1) -- pre-aligned above
                        if not pd.notna(val):
                            continue
                        raw_w = G[ticker][nb].get("weight", 1.0)
                        inv_w = 1.0 / max(raw_w, 1e-8)
                        vals.append(float(val))
                        weights.append(inv_w)

                    if vals:
                        w_arr = np.array(weights)
                        rec[feat_key] = float(np.dot(w_arr, vals) / w_arr.sum())
                    else:
                        rec[feat_key] = np.nan

                recs[ticker] = rec

            snap_records[date] = recs
            # Update previous-neighbour sets for next snapshot
            for ticker in recs:
                if ticker in G.nodes:
                    prev_neighbors[ticker] = set(G.neighbors(ticker))

        # ── Stitch snapshot records into per-ticker DataFrames ─────────
        result: Dict[str, pd.DataFrame] = {}
        for ticker, df in data_dict.items():
            df_out = df.copy()

            rows = [
                {"_date": date,
                 **{col: snap_records.get(date, {}).get(ticker, {}).get(col, np.nan)
                    for col in all_net_cols}}
                for date in snap_dates
            ]
            snap_df = pd.DataFrame(rows).set_index("_date")
            snap_df.index = pd.to_datetime(snap_df.index)

            # Reindex to the ticker's full date grid and forward-fill between
            # rebuild dates.  Rows before the first snapshot remain NaN.
            full_idx = df_out.index.union(snap_df.index).sort_values()
            snap_aligned = snap_df.reindex(full_idx).ffill().reindex(df_out.index)

            for col in all_net_cols:
                df_out[col] = (
                    snap_aligned[col].values
                    if col in snap_aligned.columns
                    else np.nan
                )

            result[ticker] = df_out

        return result

    # ------------------------------------------------------------------
    # Convenience: fit_transform and CV slicing
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        data_dict: Dict[str, pd.DataFrame],
        y=None,
        **fit_params,
    ) -> Dict[str, pd.DataFrame]:
        """fit() then transform() in one call -- the recommended offline step."""
        return self.fit(data_dict).transform(data_dict)

    @staticmethod
    def slice_features(
        transformed: Dict[str, pd.DataFrame],
        idx: pd.Index,
    ) -> Dict[str, pd.DataFrame]:
        """
        Slice the output of transform() to a date index.

        Use this inside the CV loop to split pre-computed network features
        into train and test folds without re-fitting any graphs.

        Parameters
        ----------
        transformed : output of transform() (or fit_transform())
        idx         : DatetimeIndex (or any pandas Index) of dates to keep.
                      Dates not present in a ticker's DataFrame are silently
                      ignored.

        Returns
        -------
        Dict[str, DataFrame] -- same keys, DataFrames restricted to *idx*.

        Example
        -------
            full = net.fit_transform(data_dict)
            for train_idx, test_idx in folds:
                train = FinanceNetworkBase.slice_features(full, train_idx)
                test  = FinanceNetworkBase.slice_features(full, test_idx)
        """
        result = {}
        for ticker, df in transformed.items():
            shared = df.index.intersection(idx)
            if not shared.empty:
                result[ticker] = df.loc[shared]
        return result

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @property
    def n_snapshots_(self) -> int:
        """Number of graph snapshots built during fit()."""
        return len(self._snapshots)

    def snapshot_degrees(self) -> pd.DataFrame:
        """
        Return a (n_snapshots x n_tickers) DataFrame of node degrees.

        Useful for visualising graph dynamics: sudden degree spikes or drops
        across the universe indicate structural breaks in the correlation
        network and often precede volatility regime changes.
        Missing entries (ticker absent from a snapshot) are NaN.
        """
        rows = []
        for date, G in self._snapshots:
            row = {t: float(G.degree(t)) for t in G.nodes}
            row["_date"] = date
            rows.append(row)
        df = pd.DataFrame(rows).set_index("_date")
        df.index = pd.to_datetime(df.index)
        return df

    def snapshot_edge_counts(self) -> pd.Series:
        """
        Return a Series of edge counts per snapshot date.

        A sharp drop in edges signals the threshold became binding (threshold
        graph) or distances all equalised -- both are worth investigating.
        """
        return pd.Series(
            {date: G.number_of_edges() for date, G in self._snapshots},
            name="n_edges",
        )


# ---------------------------------------------------------------------------
# Concrete metric subclasses
# ---------------------------------------------------------------------------

class SquaredCorrelationNetwork(FinanceNetworkBase):
    """
    d_ij = 1 - rho_ij^2   (squared Pearson correlation distance).

    Symmetric w.r.t. the sign of correlation: a strongly anti-correlated pair
    (rho = -0.9) is just as close as a strongly positively correlated one
    (rho = +0.9), which is appropriate when both represent structural linkage.

    Equivalent to 1 - R^2: the fraction of variance in stock j not explained
    by a linear regression on stock i.

    For graph_type="threshold", threshold=0.2 connects pairs where rho^2 > 0.8
    (very tightly co-moving stocks only).
    """

    def _compute_distance_matrix(self, returns_window: pd.DataFrame) -> pd.DataFrame:
        tickers = returns_window.columns.tolist()
        dist = _squared_corr_distance(returns_window.values.astype(float))
        return pd.DataFrame(dist, index=tickers, columns=tickers)


class PartialCorrelationNetwork(FinanceNetworkBase):
    """
    d_ij = 1 - pcorr_ij^2   (squared partial correlation distance).

    Partial correlations condition out all other stocks, so an edge between i
    and j indicates a *direct* dependency -- not one mediated by a common
    factor.  The resulting graph is typically sparser and more interpretable
    than the full-correlation graph.

    The correlation matrix is regularised with diagonal shrinkage before
    precision inversion.  Shrinkage is automatically increased when the window
    is shorter than the number of tickers (under-determined regime).

    Parameters
    ----------
    shrinkage : float in (0, 1)
        Ledoit-Wolf-style diagonal regularisation strength.  Increase if the
        precision matrix is ill-conditioned or if n_obs is close to n_tickers.
    """

    def __init__(self, *args, shrinkage: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.shrinkage = shrinkage

    def _compute_distance_matrix(self, returns_window: pd.DataFrame) -> pd.DataFrame:
        tickers = returns_window.columns.tolist()
        dist = _partial_corr_distance(
            returns_window.values.astype(float),
            shrinkage=self.shrinkage,
        )
        return pd.DataFrame(dist, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Module-level alias for slice_features (convenience import)
# ---------------------------------------------------------------------------

def slice_features(
    transformed: Dict[str, pd.DataFrame],
    idx: pd.Index,
) -> Dict[str, pd.DataFrame]:
    """
    Module-level alias for FinanceNetworkBase.slice_features().
    Import directly for a cleaner CV loop::

        from models.correlation_network import slice_features
        train = slice_features(full_features, train_idx)
        test  = slice_features(full_features, test_idx)
    """
    return FinanceNetworkBase.slice_features(transformed, idx)