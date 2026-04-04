"""
network_models.py
=================
Network-augmented volatility models.

All models in this file expect that the per-ticker DataFrames have already
been enriched with net_* columns by ``FinanceNetworkBase.fit_transform()``
before cross-validation begins.  See ``models/correlation_network.py`` for
the offline build pattern.

Models
------
NetworkHARRegressor
    Single-stage Lasso regression using the full HAR-Extended feature set
    plus network structural features (degree, neighbourhood turnover, IDW RVs).
    Straightforward baseline for network-augmented prediction.

NetworkVARRegressor
    Two-stage "network error correction" model:
      Stage 1 -- Standard HAR-Extended OLS fit in log space (own features only).
      Stage 2 -- Ridge regression of the stage-1 log-space residuals on the
                 network features (degree, turnover, IDW RVs of neighbours).
      Prediction -- exp(stage1_log_pred + stage2_log_correction).

    Motivation: the HAR residuals contain common-factor / contagion variance
    that the own-stock lags cannot explain.  The network features capture this
    cross-sectional component.  Fitting the two stages separately prevents the
    network terms from being obscured by the much larger HAR coefficients and
    allows different regularisation strengths for each stage.

Network features expected (produced by FinanceNetworkBase.transform())
----------------------------------------------------------------------
    net_degree            node degree in the current graph snapshot
    net_degree_change     neighbourhood turnover:
                            1 - |N_{t-1} ∩ N_t| / max(1, |N_t|)
                          0 = stable, 1 = complete rewiring; NaN at t=0
    net_idw_log_RV1       IDW mean of neighbours' log_RV1
    net_idw_log_RV5       IDW mean of neighbours' log_RV5
    net_idw_log_RV22      IDW mean of neighbours' log_RV22
    net_idw_Returns       IDW mean of neighbours' signed returns

Missing net_* values (rows before the first graph snapshot) are imputed
with 0.0 --- a sensible prior when no graph information is available.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Shared feature lists
# ---------------------------------------------------------------------------

MARKET_FEATURES: List[str] = [
    "Market_Returns",
    "log_Market_RV5",
    "log_Market_RV22",
]

HAR_FEATURES: List[str] = [
    "log_RV1",
    "log_RV5",
    "log_RV10",
    "log_RV22",
    "log_neg_semi5",
    "log_pos_semi5",
    "Returns",          # signed return: captures leverage effect (negative
                        # shocks drive more future volatility than positive ones)
    "Market_Returns",   # SPY market return: captures systematic risk
    "log_Market_RV5",   # 5-day market realised variance (log)
    "log_Market_RV22",  # 22-day market realised variance (log)
]


def har_features(use_market: bool = True) -> List[str]:
    """Return HAR feature list, optionally excluding market columns."""
    if use_market:
        return list(HAR_FEATURES)
    return [f for f in HAR_FEATURES if f not in MARKET_FEATURES]

NET_FEATURES: List[str] = [
    "net_degree",
    "net_degree_change",
    "net_idw_log_RV1",
    "net_idw_log_RV5",
    "net_idw_log_RV22",
    "net_idw_Returns",  # IDW mean of neighbours' signed returns: cross-sectional
                        # leverage signal -- negative neighbour returns predict
                        # higher own future volatility via contagion
]

# Extended network features (clustering + market-wide connectivity)
CLUSTERING_FEATURES: List[str] = [
    "net_node_clustering",
    "net_global_clustering",
    "net_avg_abs_corr",
]

# All network features combined
NET_FEATURES_FULL: List[str] = NET_FEATURES + CLUSTERING_FEATURES

# Sign-split IDW features: each IDW metric is split into positive-correlation
# and negative-correlation neighbour groups based on the sign of rho_ij.
# This doubles the IDW parameters but captures asymmetric contagion effects:
# positively-correlated neighbours transmit co-movement shocks, while
# negatively-correlated neighbours provide diversification / hedging signals.
SIGN_SPLIT_IDW_FEATURES: List[str] = [
    "net_idw_pos_log_RV1",
    "net_idw_neg_log_RV1",
    "net_idw_pos_log_RV5",
    "net_idw_neg_log_RV5",
    "net_idw_pos_log_RV22",
    "net_idw_neg_log_RV22",
    "net_idw_pos_Returns",
    "net_idw_neg_Returns",
]

# Sign-split network features (degree + sign-split IDW, no combined IDW)
NET_FEATURES_SIGN_SPLIT: List[str] = [
    "net_degree",
    "net_degree_change",
] + SIGN_SPLIT_IDW_FEATURES

NET_FEATURES_SIGN_SPLIT_FULL: List[str] = (
    NET_FEATURES_SIGN_SPLIT + CLUSTERING_FEATURES
)

# Master list of ALL possible net_ columns (for _fill_net)
_ALL_NET_COLUMNS: List[str] = list(dict.fromkeys(
    NET_FEATURES_FULL + SIGN_SPLIT_IDW_FEATURES
))

# Feature columns used by per-rank neighbor features (must match the
# feature_cols parameter in the FinanceNetworkBase constructor).
_RANK_FEATURE_COLS: List[str] = ["log_RV1", "log_RV5", "log_RV22", "Returns"]


def _knn_rank_features(k: int) -> List[str]:
    """Return per-rank neighbour feature column names for a given k."""
    return (
        [f"net_nn{r}_{c}" for r in range(k) for c in _RANK_FEATURE_COLS]
        + [f"net_nn{r}_dist" for r in range(k)]
    )


def _fill_net(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN in network columns with 0.0.

    NaNs appear in rows before the first graph snapshot (no connectivity
    information yet).  Using 0 rather than mean imputation avoids introducing
    test-period statistics into training rows.
    """
    net_cols_present = [c for c in X.columns if c.startswith("net_")]
    if net_cols_present:
        X = X.copy()
        X[net_cols_present] = X[net_cols_present].fillna(0.0)
    return X


def _select_net_features(
    use_clustering: bool,
    use_sign_split: bool = False,
) -> List[str]:
    """Return the appropriate network feature list."""
    if use_sign_split:
        base = NET_FEATURES_SIGN_SPLIT_FULL if use_clustering else NET_FEATURES_SIGN_SPLIT
    else:
        base = NET_FEATURES_FULL if use_clustering else NET_FEATURES
    return base


def _dedupe_preserve_order(columns: List[str]) -> List[str]:
    """Remove duplicate column names while preserving their first appearance."""
    return list(dict.fromkeys(columns))


# ---------------------------------------------------------------------------
# NetworkHARRegressor  (single-stage)
# ---------------------------------------------------------------------------

class NetworkHARRegressor(BaseEstimator, RegressorMixin):
    """
    Single-stage log-space Lasso regressing on HAR + network features.

    log_Y ~ log_RV1 + log_RV5 + log_RV10 + log_RV22
            + log_neg_semi5 + log_pos_semi5
            + net_degree + net_degree_change
            + net_idw_log_RV1 + net_idw_log_RV5 + net_idw_log_RV22

    A per-fold StandardScaler (embedded in a Pipeline) normalises all inputs
    before regularised regression, ensuring scale-invariance and no leakage.

    Parameters
    ----------
    lasso_alpha : float
        L1 penalty strength.  Lasso naturally zeroes out irrelevant network
        features when they add no predictive value.  Set to 0.0 for OLS.
    ridge_alpha : float
        L2 penalty.  Used when lasso_alpha == 0.0 and ridge_alpha > 0.0.
    """

    def __init__(
        self,
        lasso_alpha: float = 0.05,
        ridge_alpha: float = 0.0,
        use_clustering: bool = False,
        use_sign_split: bool = False,
        use_market: bool = True,
    ):
        self.lasso_alpha = lasso_alpha
        self.ridge_alpha = ridge_alpha
        self.use_clustering = use_clustering
        self.use_sign_split = use_sign_split
        self.use_market = use_market
        net_feats = _select_net_features(use_clustering, use_sign_split)
        self.features: List[str] = har_features(use_market) + net_feats
        self._pipe: Optional[Pipeline] = None

    def _make_regressor(self):
        if self.lasso_alpha > 0:
            return Lasso(alpha=self.lasso_alpha, max_iter=20_000)
        if self.ridge_alpha > 0:
            return Ridge(alpha=self.ridge_alpha)
        return LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NetworkHARRegressor":
        X_fit = _fill_net(X)[self.features]
        self._pipe = Pipeline(
            [("scaler", StandardScaler()), ("reg", self._make_regressor())]
        )
        self._pipe.fit(X_fit, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_pred = _fill_net(X)[self.features]
        return np.exp(self._pipe.predict(X_pred))


# ---------------------------------------------------------------------------
# NetworkVARRegressor  (two-stage network error-correction)
# ---------------------------------------------------------------------------

class NetworkVARRegressor(BaseEstimator, RegressorMixin):
    """
    Two-stage Network Vector AutoRegression for weekly realised variance.

    Motivation
    ----------
    Standard HAR models only use each stock's own past volatility.  In a
    connected market, shocks propagate through the correlation network, so
    a stock's HAR residuals should be systematically related to its neighbours'
    volatility state.  This model captures that cross-sectional contagion term
    explicitly by fitting it in a separate stage.

    Stage 1 -- Own dynamics
        Fit a HAR-Extended OLS regression in log space using only own lags:
            log_Ŷ_1 = f(log_RV1, log_RV5, log_RV10, log_RV22,
                         log_neg_semi5, log_pos_semi5, Returns)
        Compute training residuals: e_t = log_Y_t - log_Ŷ_1,t

    Stage 2 -- Network error correction
        Fit a Ridge regression of e_t on the network features:
            ê_t = g(net_degree, net_degree_change,
                    net_idw_log_RV1, net_idw_log_RV5, net_idw_log_RV22)
        The correction is bounded to [-bound, +bound] in log space to prevent
        explosive predictions from noisy network features.

    Final prediction
        log_Ŷ = log_Ŷ_1 + clamp(ê, -bound, +bound)
        Ŷ     = exp(log_Ŷ)

    Parameters
    ----------
    stage2_alpha : float
        Ridge penalty for the network error-correction stage.  Higher values
        shrink the network correction towards zero.  Set to 0.0 for an
        unregularised stage-2 OLS fit.
    correction_bound : Optional[float]
        Maximum absolute log-space correction from stage 2.  Acts as a safety
        valve against extreme network predictions.  A value of 0.5 means the
        network can shift the HAR prediction by at most e^0.5 ≈ 1.65× up or
        down in original scale.  Set to None to disable clipping entirely.
    """

    def __init__(
        self,
        stage2_alpha: float = 0.1,
        correction_bound: Optional[float] = 0.5,
        use_clustering: bool = False,
        use_sign_split: bool = False,
        use_market: bool = True,
    ):
        self.stage2_alpha = stage2_alpha
        self.correction_bound = correction_bound
        self.use_clustering = use_clustering
        self.use_sign_split = use_sign_split
        self.use_market = use_market
        self._har_feats: List[str] = har_features(use_market)
        self._net_feats: List[str] = _select_net_features(use_clustering, use_sign_split)
        self.features: List[str] = self._har_feats + self._net_feats

        self._stage1: Optional[Pipeline] = None
        self._stage2: Optional[Pipeline] = None

    # ── Stage 1: own HAR-Extended OLS ──────────────────────────────────

    def _fit_stage1(self, X: pd.DataFrame, log_y: pd.Series) -> pd.Series:
        """Fit stage-1 OLS; return in-sample log-space residuals."""
        self._stage1 = Pipeline(
            [("scaler", StandardScaler()), ("reg", LinearRegression())]
        )
        self._stage1.fit(X[self._har_feats], log_y)
        log_pred1 = pd.Series(
            self._stage1.predict(X[self._har_feats]), index=log_y.index
        )
        return log_y - log_pred1   # residuals

    # ── Stage 2: network residual correction ───────────────────────────

    def _fit_stage2(self, X: pd.DataFrame, residuals: pd.Series) -> None:
        """Fit stage-2 Ridge on network features using HAR residuals as target."""
        X_net = _fill_net(X)[self._net_feats]
        stage2_reg = Ridge(alpha=self.stage2_alpha) if self.stage2_alpha > 0 else LinearRegression()
        self._stage2 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", stage2_reg),
            ]
        )
        self._stage2.fit(X_net, residuals)

    # ── sklearn API ────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NetworkVARRegressor":
        log_y = y.copy()   # y is already log_Y (see fit_predict_model)

        residuals = self._fit_stage1(X, log_y)
        self._fit_stage2(X, residuals)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_pred1 = self._stage1.predict(X[self._har_feats])

        X_net = _fill_net(X)[self._net_feats]
        correction = self._stage2.predict(X_net)
        if self.correction_bound is not None:
            correction = np.clip(correction, -self.correction_bound, self.correction_bound)

        return np.exp(log_pred1 + correction)


# ---------------------------------------------------------------------------
# LearnedWeightNetworkHARRegressor  (learned m×k projection)
# ---------------------------------------------------------------------------

class LearnedWeightNetworkHARRegressor(BaseEstimator, RegressorMixin):
    """
    Network model with a learned m×k weight matrix for neighbour aggregation.

    Instead of using fixed IDW weights, this model sorts each ticker's k
    nearest neighbours by ascending distance and applies a **shared** learned
    weight matrix W (m × k) across all feature columns.  This projects the
    k per-neighbour values for each feature into m < k/2 compressed features,
    capturing the most predictive distance-rank patterns.

    The weight matrix W is learned from training data via truncated SVD of the
    stacked neighbour feature matrix, so the m retained components explain the
    most variance in the neighbour structure.  The projected features are then
    concatenated with HAR features for a final Ridge/Lasso regression.

    Expected input columns (produced by FinanceNetworkBase.transform()):
        net_nn{r}_{col}  for r in 0..k-1, col in feature_cols
        net_nn{r}_dist   for r in 0..k-1
    Plus the standard HAR features.

    Parameters
    ----------
    k : int
        Number of nearest neighbours (must match the graph's k).
    m : int
        Projection dimension.  Must satisfy m < k / 2.
    alpha : float
        Ridge penalty for the final regression stage.
    lasso_alpha : float
        If > 0, use Lasso instead of Ridge for final regression.
    use_clustering : bool
        Include global clustering features (net_node_clustering, etc.).
    """

    def __init__(
        self,
        k: int = 5,
        m: int = 2,
        alpha: float = 1.0,
        lasso_alpha: float = 0.0,
        use_clustering: bool = False,
        use_market: bool = True,
    ):
        self.k = k
        self.m = m
        self.alpha = alpha
        self.lasso_alpha = lasso_alpha
        self.use_clustering = use_clustering
        self.use_market = use_market

        # Build feature list
        _har = har_features(use_market)
        rank_feats = _knn_rank_features(k)
        struct_feats = ["net_degree", "net_degree_change"]
        if use_clustering:
            struct_feats += CLUSTERING_FEATURES
        self.features: List[str] = _dedupe_preserve_order(
            _har + struct_feats + rank_feats
        )

        self._W: Optional[np.ndarray] = None       # (m, k) projection matrix
        self._pipe: Optional[Pipeline] = None       # final regressor
        self._feat_cols: List[str] = _RANK_FEATURE_COLS
        self._har_and_struct: List[str] = _dedupe_preserve_order(
            _har + struct_feats
        )

    def _extract_nn_tensor(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extract the (n_samples, n_feat_cols, k) tensor of per-rank neighbour
        values from the DataFrame.
        """
        n = len(X)
        n_f = len(self._feat_cols)
        k = self.k
        tensor = np.zeros((n, n_f, k))
        for fi, fc in enumerate(self._feat_cols):
            for r in range(k):
                col = f"net_nn{r}_{fc}"
                if col in X.columns:
                    tensor[:, fi, r] = X[col].values
        return tensor

    def _learn_W(self, nn_tensor: np.ndarray) -> np.ndarray:
        """
        Learn the m×k weight matrix via truncated SVD of the stacked
        neighbour feature matrix.

        The stacked matrix has shape (n_samples * n_feat_cols, k).
        The top-m right singular vectors form the rows of W.
        """
        n, n_f, k = nn_tensor.shape
        stacked = nn_tensor.reshape(n * n_f, k)
        # Center columns before SVD
        col_means = stacked.mean(axis=0)
        stacked_c = stacked - col_means
        # Truncated SVD: only need top-m right singular vectors
        try:
            _, _, Vt = np.linalg.svd(stacked_c, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback: use identity-like matrix
            W = np.zeros((self.m, k))
            for i in range(min(self.m, k)):
                W[i, i] = 1.0
            return W
        return Vt[:self.m]  # (m, k)

    def _project(self, nn_tensor: np.ndarray) -> np.ndarray:
        """
        Apply W to the neighbour tensor.

        Returns shape (n_samples, n_feat_cols * m).
        """
        # Z[i, f, :] = W @ nn_tensor[i, f, :]   shape (m,)
        Z = np.einsum("mk,nfk->nfm", self._W, nn_tensor)
        return Z.reshape(len(nn_tensor), -1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LearnedWeightNetworkHARRegressor":
        X_filled = _fill_net(X)

        # Extract neighbour tensor and learn W
        nn_tensor = self._extract_nn_tensor(X_filled)
        self._W = self._learn_W(nn_tensor)

        # Project and concatenate with HAR + structural features
        Z = self._project(nn_tensor)
        X_har = X_filled[self._har_and_struct].values
        X_full = np.hstack([X_har, Z])

        # Final regression
        if self.lasso_alpha > 0:
            reg = Lasso(alpha=self.lasso_alpha, max_iter=20_000)
        elif self.alpha > 0:
            reg = Ridge(alpha=self.alpha)
        else:
            reg = LinearRegression()

        self._pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        self._pipe.fit(X_full, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_filled = _fill_net(X)
        nn_tensor = self._extract_nn_tensor(X_filled)
        Z = self._project(nn_tensor)
        X_har = X_filled[self._har_and_struct].values
        X_full = np.hstack([X_har, Z])
        return np.exp(self._pipe.predict(X_full))
